#!/usr/bin/env bash
# RF-DETR on Cityscapes: detect (medium) + segment (seg-medium), in sequence.
#
# Size picked by ENGINE-ONLY latency (e2e is meaningless for sizing here - rfdetr's shipped path
# does a PIL resize of the full 2048x1024 frame, so every variant lands at 8-16 ms e2e regardless
# of the network). Engine-only, vs D-FINE-seg-S:
#     detect   D-FINE 1.58 | rfdetr nano .384 1.11 · small .512 1.49 · MEDIUM .576 1.62
#     segment  D-FINE 2.09 | rfdetr seg-nano 1.44 · seg-small 1.72 · SEG-MEDIUM .432 1.95
# medium / seg-medium are the LARGEST variants that fit D-FINE's budget => the most generous
# fair pick for the rival. Each keeps its own default resolution (576 / 432).
#
# Epochs: 100 budget + rfdetr's OWN early stopping (patience 15, on EMA val). RF-DETR is a bigger
# model and converges in fewer epochs, so rather than guess 55 vs 100 we give it the generous
# budget and let it stop itself once val plateaus - no wasted compute, no risk of handicapping it
# with a short run. Best (EMA) checkpoint is saved regardless, so overshooting the peak is harmless.
set -u
cd "$(dirname "$0")/.."  # repo root: python -m round1_640.<script>

EPOCHS=${EPOCHS:-100}
ROOT="${CITYSCAPES_ROOT:-$(cd .. && pwd)}"  # workspace that holds data/ and output/ (paths.py)
LOGS="$ROOT/output/logs"
mkdir -p "$LOGS"

for v in medium seg-medium; do
  echo "===================================================================="
  echo "  RF-DETR $v   epochs=$EPOCHS   $(date)"
  echo "===================================================================="
  ./.venv-rfdetr/bin/python -m round1_640.train_rfdetr --variant "$v" --epochs "$EPOCHS" \
      2>&1 | tee "$LOGS/rfdetr_${v}.log"
  rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then
    echo "!!! RF-DETR $v FAILED (rc=$rc) - continuing"
  else
    echo ">>> RF-DETR $v done  $(date)"
  fi
done

echo "ALL RF-DETR TRAINING DONE $(date)"
