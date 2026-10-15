"""Reproduction gate for YOLO26-sem: re-run ultralytics' OWN val and match the stored mIoU.

The docs table (78.3 ... 83.6) is the training-time val stored in each checkpoint
(`train_metrics["metrics/mIoU"]`). The checkpoints were trained with ultralytics 8.4.50, whose
semantic val scored at NATIVE 1024x2048 (short side -> imgsz=1024). In the 8.4.92 we run, the val
loader resizes the LONG side to imgsz instead, so `imgsz=1024` would score at 512x1024 against
downsampled GT (-6 mIoU on n). `imgsz=2048, rect=True` restores the native protocol and
reproduces the stored number (n: 0.78255 vs 0.78259).

  .venv-yolo/bin/python repro_yolo_val.py [--sizes n s m l x]
"""

import argparse

import torch
from tabulate import tabulate
from ultralytics import YOLO

from paths import DATA, RIV

DATA_YAML = DATA / "yolo_sem.yaml"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", default=["n", "s", "m", "l", "x"])
    ap.add_argument("--imgsz", type=int, default=2048, help="long side; 2048 = native 1024x2048")
    args = ap.parse_args()

    rows = []
    for s in args.sizes:
        pt = RIV / f"yolo26{s}-sem.pt"
        ck = torch.load(pt, map_location="cpu", weights_only=False)
        stored = ck["train_metrics"]["metrics/mIoU"]
        # the trainer validates the EMA in half precision with rect=True
        r = YOLO(str(pt)).val(
            data=str(DATA_YAML),
            imgsz=args.imgsz,
            rect=True,
            batch=8,
            half=True,
            device=0,
            plots=False,
            verbose=False,
        )
        ours = r.results_dict["metrics/mIoU"]
        rows.append(
            {
                "model": f"YOLO26{s}-sem",
                "trained_with": ck.get("version"),
                "stored_mIoU": stored,
                "repro_mIoU": round(ours, 5),
                "delta": round(ours - stored, 5),
            }
        )
    print(tabulate(rows, headers="keys", tablefmt="github"))
