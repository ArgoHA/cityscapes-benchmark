"""Run YOLO26 through ULTRALYTICS' OWN shipped TensorRT inference and dump predictions.

No custom pre/postprocessing: `YOLO(<engine>).predict()` goes through ultralytics' AutoBackend
TensorRTBackend + their own predictor (LetterBox -> engine -> NMS-free decode -> scale back to
original image coords). We only adapt the returned `Results` object into the dump format.

Engine is produced by ultralytics' own exporter (`export(format="engine", half=True)`), which
builds with the venv's TensorRT 10.13.3.9 - the same version D-FINE's engines use.

  .venv-yolo/bin/python infer_yolo.py --weights <run>/weights/best.pt --task detect
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

import dump

ROOT = Path("/home/argo/Desktop/Projects/cityscapes")
VAL_CSV = ROOT / "data" / "dataset" / "val.csv"
IMAGES = ROOT / "data" / "dataset" / "images"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)  # .pt (exported to .engine) or .engine
    ap.add_argument("--task", required=True, choices=["detect", "segment", "semantic"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0, help="latency-calibration: first N images only")
    # ultralytics' native-resolution mask option: protos are upsampled straight to the original
    # image size (the same one-shot upsample D-FINE does), instead of to letterbox space and then
    # resampled again. Costs latency, which we measure.
    ap.add_argument("--retina-masks", type=int, default=1)
    args = ap.parse_args()

    w = Path(args.weights)
    if w.suffix == ".pt":  # export with ultralytics' own TRT exporter (reuse if already built)
        cached = w.with_suffix(".engine")
        engine = (
            str(cached)
            if cached.exists()
            else YOLO(str(w)).export(format="engine", imgsz=args.imgsz, half=True, device=0)
        )
    else:
        engine = str(w)
    print(f"engine: {engine}")

    model = YOLO(engine, task=args.task)
    stems = [ln.strip() for ln in VAL_CSV.read_text().splitlines() if ln.strip()]
    images = [IMAGES / s for s in stems]
    if args.limit:
        images = images[: args.limit]

    kw = dict(imgsz=args.imgsz, conf=args.conf, device=0, verbose=False)
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

    out_dir = Path(args.out or ROOT / "output" / "compare" / args.name)
    dump.run(
        args.name,
        args.task if args.task != "semantic" else "sem_seg",
        infer,
        extract,
        images,
        lambda p: cv2.imread(str(p), cv2.IMREAD_COLOR),
        out_dir,
        extra={"engine": str(engine), "conf": args.conf, "imgsz": args.imgsz,
               "retina_masks": bool(args.retina_masks) if args.task == "segment" else None,
               "ul_speed_mean": {
                   k: round(sum(s[k] for s in speeds[1:]) / max(len(speeds) - 1, 1), 2)
                   for k in ("preprocess", "inference", "postprocess")
               } if speeds else None},
    )
