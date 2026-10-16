"""Run YOLO26 through ULTRALYTICS' OWN shipped TensorRT inference and dump predictions.

No custom pre/postprocessing: `YOLO(<engine>).predict()` goes through ultralytics' AutoBackend
TensorRTBackend + their own predictor (LetterBox -> engine -> NMS-free decode -> scale back to
original image coords). We only adapt the returned `Results` object into the dump format.

Engine is produced by ultralytics' own exporter (`export(format="engine", half=True)`), which
builds with the venv's TensorRT 10.13.3.9 - the same version D-FINE's engines use.

  .venv-yolo/bin/python -m infer.infer_yolo --weights <run>/weights/best.pt --task detect

Native-res semantic (exports into --engine-dir, never next to the .pt, so an engine built at another
resolution can't be picked up by accident):

  .venv-yolo/bin/python -m infer.infer_yolo --weights yolo26x-sem.pt --task semantic --name YOLO26-x \
      --imgsz 1024 2048 --engine-dir artifacts/yolo/yolo26x-sem_1024x2048
"""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from common import dump
from common.model_stats import model_stats
from common.paths import ROOT

VAL_CSV = ROOT / "data" / "dataset" / "val.csv"
IMAGES = ROOT / "data" / "dataset" / "images"


def resolve_engine(w: Path, imgsz, engine_dir: str | None) -> str:
    """Export with ultralytics' own TRT exporter, reusing a cached engine if one exists."""
    if engine_dir is None:  # round-1 behaviour: engine next to the .pt
        cached = w.with_suffix(".engine")
        if cached.exists():
            return str(cached)
        return YOLO(str(w)).export(format="engine", imgsz=imgsz, half=True, device=0)
    d = Path(engine_dir)
    cached = d / f"{w.stem}.engine"
    if cached.exists():
        return str(cached)
    d.mkdir(parents=True, exist_ok=True)
    pt = d / w.name
    shutil.copy2(w, pt)  # the exporter writes .onnx/.engine next to its input
    return YOLO(str(pt)).export(format="engine", imgsz=imgsz, half=True, device=0, batch=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)  # .pt (exported to .engine) or .engine
    ap.add_argument("--task", required=True, choices=["detect", "segment", "semantic"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, nargs="+", default=[640], help="S or H W")
    ap.add_argument("--engine-dir", default=None, help="export/cache dir (keyed by resolution)")
    ap.add_argument(
        "--backend",
        default="trt",
        choices=["trt", "torch"],
        help="torch = the .pt through the same predictor (TRT-vs-torch gate)",
    )
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0, help="latency-calibration: first N images only")
    # ultralytics' native-resolution mask option: protos are upsampled straight to the original
    # image size (the same one-shot upsample D-FINE does), instead of to letterbox space and then
    # resampled again. Costs latency, which we measure.
    ap.add_argument("--retina-masks", type=int, default=1)
    args = ap.parse_args()
    imgsz = args.imgsz[0] if len(args.imgsz) == 1 else tuple(args.imgsz)
    hw = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz

    w = Path(args.weights)
    if args.backend == "torch":
        engine = str(w)
    elif w.suffix == ".pt":
        engine = resolve_engine(w, imgsz, args.engine_dir)
    else:
        engine = str(w)
    print(f"model: {engine}")

    model = YOLO(engine, task=args.task)
    if args.task == "semantic":
        images = dump.sem_val_images(args.limit)
    else:
        stems = [ln.strip() for ln in VAL_CSV.read_text().splitlines() if ln.strip()]
        images = [IMAGES / s for s in stems][: args.limit or None]

    kw = dict(imgsz=imgsz, conf=args.conf, device=0, verbose=False)
    if args.task == "segment":
        kw["retina_masks"] = bool(args.retina_masks)

    speeds = []  # ultralytics' own preprocess/inference/postprocess breakdown, for reporting

    def infer(img):  # timed: ultralytics' full end-to-end path
        r = model.predict(source=img, **kw)[0]
        speeds.append(dict(r.speed))
        return r

    def extract(r):  # untimed: D2H, matching bench.py's post-clock conversion
        if args.task == "semantic":
            # SemanticMask is a BaseTensor wrapper: .numpy() returns another SemanticMask,
            # so go through .data for the raw tensor.
            return {"sem_seg": r.semantic_mask.data.cpu().numpy().astype(np.uint8)}
        out = {
            "boxes": r.boxes.xyxy.cpu().numpy(),  # already original-image coords
            "scores": r.boxes.conf.cpu().numpy(),
            "labels": r.boxes.cls.cpu().numpy(),
        }
        if args.task == "segment":
            m = r.masks
            out["masks"] = (
                m.data.cpu().numpy().astype(np.uint8)
                if m is not None
                else np.zeros((0, *r.orig_shape), np.uint8)
            )
        return out

    extra = {
        "engine": str(engine),
        "conf": args.conf,
        "imgsz": imgsz,
        "backend": args.backend,
        "input": f"{hw[0]}x{hw[1]}",
        "retina_masks": bool(args.retina_masks) if args.task == "segment" else None,
    }
    if w.suffix == ".pt":  # count on the fused torch model, i.e. the graph that gets exported
        net = YOLO(str(w)).model.fuse().eval().cuda()
        extra.update(model_stats(net, torch.zeros(1, 3, *hw, device="cuda")))
        del net

    out_dir = Path(args.out or ROOT / "output" / "compare" / args.name)
    dump.run(
        args.name,
        args.task if args.task != "semantic" else "sem_seg",
        infer,
        extract,
        images,
        lambda p: cv2.imread(str(p), cv2.IMREAD_COLOR),
        out_dir,
        extra=extra,
    )
    if speeds:  # skip the first (cold) call, as dump.run does for latency
        n = max(len(speeds) - 1, 1)
        ul = {
            k: round(sum(s[k] for s in speeds[1:]) / n, 2)
            for k in ("preprocess", "inference", "postprocess")
        }
        print(f"ultralytics' own speed breakdown (ms): {ul}")
