"""Run D-FINE-seg through ITS OWN shipped TensorRT inference and dump predictions.

Uses src/infer/trt_model.py::TRT_model - the wrapper D-FINE ships and that `make bench` uses -
so D-FINE goes through the same dump/timing path as the rivals and every model is scored by the
same code from the same artifacts.

  uv run --project /home/argo/Desktop/Projects/D-FINE-seg python infer_dfine.py \
      --engine <model_dir>/model.engine --task detect --name dfine_s_det
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

DFINE = Path("/home/argo/Desktop/Projects/D-FINE-seg")
ROOT = Path("/home/argo/Desktop/Projects/cityscapes")
sys.path.insert(0, str(DFINE))
sys.path.insert(0, str(ROOT / "cityscapes-benchmark"))

from src.infer.trt_model import TRT_model  # noqa: E402

import dump  # noqa: E402

VAL_CSV = ROOT / "data" / "dataset" / "val.csv"
IMAGES = ROOT / "data" / "dataset" / "images"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--task", required=True, choices=["detect", "segment", "sem_seg"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0, help="latency-calibration: first N images only")
    args = ap.parse_args()

    n_out = 19 if args.task == "sem_seg" else 8
    model = TRT_model(
        model_path=args.engine,
        n_outputs=n_out,
        conf_thresh=args.conf,
        rect=False,
        keep_ratio=False,
        apply_nms=args.task != "sem_seg",
    )

    def infer(img):  # timed: D-FINE's own end-to-end wrapper
        return model(img)[0]

    def extract(r):  # untimed: D2H, exactly as bench.py does it after stopping the clock
        if args.task == "sem_seg":
            return {"sem_seg": r["sem_seg"].cpu().numpy().astype(np.uint8)}
        out = {
            "boxes": r["boxes"].cpu().numpy(),
            "scores": r["scores"].cpu().numpy(),
            "labels": r["labels"].cpu().numpy(),
        }
        if "masks" in r:
            out["masks"] = r["masks"].cpu().numpy().astype(np.uint8)
        return out

    stems = [ln.strip() for ln in VAL_CSV.read_text().splitlines() if ln.strip()]
    images = [IMAGES / s for s in stems]
    if args.limit:
        images = images[: args.limit]

    dump.run(
        args.name,
        args.task,
        infer,
        extract,
        images,
        lambda p: cv2.imread(str(p), cv2.IMREAD_COLOR),
        Path(args.out or ROOT / "output" / "compare" / args.name),
        extra={"engine": args.engine, "conf": args.conf},
    )
