"""Train RF-DETR on cityscapes (detect / segment).

RF-DETR has no semantic-segmentation task - that table row is D-FINE vs YOLO26 only.

Library defaults everywhere except: dataset, epochs, output dir, seed (plus the seg-medium
micro-batch, see run_rfdetr_seg.sh). In particular the model's DEFAULT square resolution is kept
(Nano 384 / Small 512 / Medium 576; Seg-S 384 / Seg-M 432) - resolution is how RF-DETR buys its
speed, so overriding it would misrepresent the model. Size is chosen to match D-FINE's measured
engine-only TRT latency instead (measure_engine.py; see PROTOCOL.md section 4).

RF-DETR reads the YOLO-format dataset directly ({train,valid}/{images,labels} + data.yaml) and
rasterizes polygon labels into masks for the segmentation head.

  .venv-rfdetr/bin/python train_rfdetr.py --variant small     --epochs 55
  .venv-rfdetr/bin/python train_rfdetr.py --variant seg-small --epochs 55
"""

import argparse
from pathlib import Path

import rfdetr

DATA = Path("/home/argo/Desktop/Projects/cityscapes/data")
OUT = Path("/home/argo/Desktop/Projects/cityscapes/output/models")

# variant -> (class name, dataset dir)
VARIANTS = {
    "nano": ("RFDETRNano", "rfdetr_det"),
    "small": ("RFDETRSmall", "rfdetr_det"),
    "medium": ("RFDETRMedium", "rfdetr_det"),
    "seg-nano": ("RFDETRSegNano", "rfdetr_seg"),
    "seg-small": ("RFDETRSegSmall", "rfdetr_seg"),
    "seg-medium": ("RFDETRSegMedium", "rfdetr_seg"),
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=VARIANTS)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--early-stopping", type=int, default=1)
    ap.add_argument("--patience", type=int, default=15)
    # rfdetr's defaults are batch_size=4 x grad_accum=4 (effective 16). On Cityscapes the SEG
    # variants OOM a 16 GB card: a crowded scene makes loss_masks' target tensor ~2.7 GB
    # (criterion.py:499) and training dies mid-run. Halving the micro-batch and doubling
    # accumulation keeps the EFFECTIVE batch at their default 16, so the optimization recipe is
    # unchanged (RF-DETR is LayerNorm-based - no batch-stat dependence); only activation memory
    # halves. Detect trains fine at their default 4.
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=None)
    # infra only, not a training hyperparameter: rfdetr defaults to 2 workers and rasterizes
    # polygon masks on CPU, which starved the GPU to 28% utilization (22 min/epoch).
    ap.add_argument("--num-workers", type=int, default=None)
    args = ap.parse_args()

    cls_name, ds = VARIANTS[args.variant]
    name = args.name or f"rfdetr_{args.variant.replace('-', '_')}"

    extra = {}
    if args.batch_size:
        extra["batch_size"] = args.batch_size
    if args.grad_accum:
        extra["grad_accum_steps"] = args.grad_accum
    if args.num_workers:
        extra["num_workers"] = args.num_workers

    model = getattr(rfdetr, cls_name)()  # default resolution for this size
    model.train(
        dataset_dir=str(DATA / ds),
        epochs=args.epochs,
        output_dir=str(OUT / name),
        seed=0,
        tensorboard=True,
        **extra,
        # RF-DETR is a bigger model and converges in fewer epochs than YOLO. Rather than guess a
        # budget, give it a generous one and let rfdetr's own early stopping cut it off once val
        # stops improving - no wasted compute, and no risk of handicapping it with a short run.
        # The best (EMA) checkpoint is saved regardless, so overfitting past the peak is harmless.
        early_stopping=args.early_stopping,
        early_stopping_patience=args.patience,
        early_stopping_use_ema=True,
    )
