"""Run D-FINE-seg through ITS OWN shipped TensorRT inference and dump predictions.

Uses dfine_seg/infer/trt_model.py::TRTModel - the wrapper D-FINE ships and that `make bench` uses -
so D-FINE goes through the same dump/timing path as the rivals and every model is scored by the
same code from the same artifacts.

  uv run --project "$DFINE_SEG_ROOT" python -m infer.infer_dfine \
      --engine <model_dir>/model.engine --task detect --name dfine_s_det
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from common.paths import ROOT, dfine_root  # noqa: E402

sys.path.insert(0, str(dfine_root()))

from dfine_seg.infer.trt_model import TRTModel  # noqa: E402

from common import dump  # noqa: E402

VAL_CSV = ROOT / "data" / "dataset" / "val.csv"
IMAGES = ROOT / "data" / "dataset" / "images"


def stats_from_pt(pt: Path, hw: tuple[int, int]) -> dict:
    """Params/GMACs of the torch model the engine was exported from (same counter as the rivals)."""
    from dfine_seg.infer.torch_model import TorchModel

    from common.model_stats import model_stats

    tm = TorchModel(model_path=str(pt), input_height=hw[0], input_width=hw[1], device="cuda")
    return model_stats(tm.model.eval(), torch.zeros(1, 3, *hw, device="cuda"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--task", required=True, choices=["detect", "segment", "sem_seg"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0, help="latency-calibration: first N images only")
    ap.add_argument("--weights", default=None, help="model.pt the engine came from (params/GMACs)")
    args = ap.parse_args()

    model = TRTModel(
        model_path=args.engine,
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

    if args.task == "sem_seg":
        images = dump.sem_val_images(args.limit)
    else:
        stems = [ln.strip() for ln in VAL_CSV.read_text().splitlines() if ln.strip()]
        images = [IMAGES / s for s in stems][: args.limit or None]

    h, w = model.input_size
    extra = {"engine": args.engine, "conf": args.conf, "input": f"{h}x{w}", "backend": "trt"}
    if args.weights:
        extra.update(stats_from_pt(Path(args.weights), (h, w)))

    dump.run(
        args.name,
        args.task,
        infer,
        extract,
        images,
        lambda p: cv2.imread(str(p), cv2.IMREAD_COLOR),
        Path(args.out or ROOT / "output" / "compare" / args.name),
        extra=extra,
    )
