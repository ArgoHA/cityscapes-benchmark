"""Run RF-DETR through ROBOFLOW'S OWN shipped TensorRT inference and dump predictions.

No custom pre/postprocessing. Everything below is rfdetr's own code:
  - preprocess : torchvision Resize((res,res)) + ToImage + ToDtype + rfdetr.datasets.transforms.Normalize
                 (verbatim `rfdetr.export.benchmark.infer_transforms`, with the model's own
                 resolution instead of that helper's hardcoded 640)
  - runtime    : rfdetr.export.benchmark.TRTInference   (their TRT class, sync_mode like their main())
  - postprocess: rfdetr.models.postprocess.PostProcess  (sigmoid top-k, cxcywh->xyxy scaled to the
                 original size, masks bilinear-resized to original and thresholded) - the full
                 version that also handles masks, unlike benchmark.py's detection-only helper.

One upstream BUG worked around (import plumbing only - no model logic is touched): rfdetr 1.8.3
`export/benchmark.py` does `import _tensorrt as trt` (a typo), so `trt` is always None and
TRTInference always raises "TensorRT is not installed". We alias the module.

One version pin, NOT a bug: rfdetr's `_tensorrt.trtexec` helper shells out to whatever `trtexec` is
on PATH (system TRT 10.11 here), and TRT engines are version-locked, so the 10.13.3.9 runtime every
engine in this benchmark is pinned to cannot load what it writes. We build via the Python API at the
venv's version with their flags (fp16, 4GB workspace). Pin the venv's TRT to the system trtexec
version instead and their helper works unmodified - see `rfdetr_trt.py`.

  .venv-rfdetr/bin/python -m infer.infer_rfdetr --variant small --weights <ckpt.pth> --name rfdetr_s_det
"""

import argparse
import sys
from pathlib import Path

import tensorrt  # noqa: F401

sys.modules.setdefault("_tensorrt", tensorrt)  # upstream typo: benchmark.py imports `_tensorrt`

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision.transforms.v2 import Compose, Resize, ToDtype, ToImage  # noqa: E402

import rfdetr  # noqa: E402
from rfdetr.datasets.transforms import Normalize  # noqa: E402
from rfdetr.export.benchmark import TRTInference  # noqa: E402
from rfdetr.models.postprocess import PostProcess  # noqa: E402

from common import dump  # noqa: E402
from common.build_engine import build  # noqa: E402
from common.paths import ROOT  # noqa: E402

VAL_CSV = ROOT / "data" / "dataset" / "val.csv"
IMAGES = ROOT / "data" / "dataset" / "images"

VARIANTS = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "seg-nano": "RFDETRSegNano",
    "seg-small": "RFDETRSegSmall",
    "seg-medium": "RFDETRSegMedium",
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=VARIANTS)
    ap.add_argument("--weights", default=None)  # fine-tuned .pth; omit -> COCO checkpoint
    ap.add_argument("--name", required=True)
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0, help="latency-calibration: first N images only")
    args = ap.parse_args()

    is_seg = args.variant.startswith("seg")
    kw = {"pretrain_weights": args.weights} if args.weights else {}
    model = getattr(rfdetr, VARIANTS[args.variant])(**kw)
    res = model.model_config.resolution
    num_select = model.model_config.num_select
    print(f"{args.variant}: resolution={res} num_select={num_select}")

    out_dir = Path(args.out or ROOT / "output" / "compare" / args.name)
    onnx_dir = out_dir / "onnx"  # keep the engine WITH its dump: bench_all resolves it from there
    model.export(output_dir=str(onnx_dir), verbose=False)  # rfdetr's own ONNX exporter
    onnx = sorted(onnx_dir.glob("*.onnx"))[0]
    engine = build(onnx, onnx.with_suffix(".engine"), half=True, workspace_gb=4)

    trt_model = TRTInference(engine_path=str(engine), device="cuda:0", sync_mode=True)
    post = PostProcess(num_select=num_select)
    # rfdetr's own val/inference transform, at this model's resolution
    tf = Compose([Resize((res, res)), ToImage(), ToDtype(torch.float32, scale=True), Normalize()])

    def infer(img: Image.Image):  # timed: rfdetr's own transform + TRTInference + PostProcess
        W, H = img.size
        t, _ = tf(img, None)
        # .contiguous() is MANDATORY: rfdetr's Normalize() returns a non-contiguous tensor, and
        # their TRTInference hands .data_ptr() straight to TRT, which assumes contiguous NCHW. Feed
        # it the non-contiguous tensor and the engine silently reads scrambled memory - logits come
        # out all-negative (max -2.1 instead of +3.0), every score collapses below 0.11, and F1
        # reads ~0.00. Same values, correct layout; their pre/postprocess is untouched.
        outs = trt_model({"input": t[None].contiguous().cuda()})
        packed = {"pred_logits": outs["labels"], "pred_boxes": outs["dets"]}
        if is_seg:
            packed["pred_masks"] = outs["masks"]
        r = post(packed, torch.tensor([[H, W]], device="cuda"))[0]
        # The class head is (num_classes + 1) wide - the extra trailing slot is DETR's "no-object"
        # class. PostProcess's `topk % n_logits` can therefore emit label == num_classes, which is
        # not a real category (rfdetr's own COCO eval drops it, as its id isn't in the GT set).
        n_cls = outs["labels"].shape[-1] - 1
        # conf filter stays on the clock (GPU op) - D-FINE and ultralytics both filter inside
        # their own inference call, so RF-DETR must too.
        keep = (r["scores"] >= args.conf) & (r["labels"] < n_cls)
        r = {k: v[keep] for k, v in r.items()}
        r["_hw"] = (H, W)
        return r

    def extract(r):  # untimed: D2H, matching bench.py's post-clock conversion
        H, W = r["_hw"]
        out = {
            "boxes": r["boxes"].cpu().numpy(),
            "scores": r["scores"].cpu().numpy(),
            "labels": r["labels"].cpu().numpy(),
        }
        if is_seg:
            m = r["masks"].cpu().numpy().astype(np.uint8)
            out["masks"] = m.reshape(-1, H, W) if m.size else np.zeros((0, H, W), np.uint8)
        return out

    stems = [ln.strip() for ln in VAL_CSV.read_text().splitlines() if ln.strip()]
    images = [IMAGES / s for s in stems]
    if args.limit:
        images = images[: args.limit]

    dump.run(
        args.name,
        "segment" if is_seg else "detect",
        infer,
        extract,
        images,
        lambda p: Image.open(p).convert("RGB"),
        out_dir,
        extra={"engine": str(engine), "conf": args.conf, "resolution": res},
    )
