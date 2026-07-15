#!/usr/bin/env bash
# RF-DETR seg-medium, retry after the default recipe OOM'd.
#
# The first attempt used rfdetr's defaults (batch 4 x grad_accum 4) and died at ~epoch 22 with
# CUDA OOM inside loss_masks: a crowded Cityscapes scene makes the target-mask tensor ~2.7 GB
# (criterion.py:499) and a 16 GB card has nothing left. It had saved checkpoint_best_ema.pth from
# epoch 16, which is exactly the trap - a checkpoint from a crashed, still-improving run is NOT a
# trained model, and benching it would have understated RF-DETR.
#
# Fix: batch_size 2 x grad_accum 8 -> EFFECTIVE batch stays 16 (their default), so the optimization
# recipe is unchanged; only activation memory halves. RF-DETR is LayerNorm-based, so there is no
# batch-statistic dependence to worry about.
# Also num_workers 2 -> 8: pure infra. Their default starved the GPU to 28% util (22 min/epoch)
# because polygon->mask rasterization runs on CPU.
#
# On success we touch TRAINING_COMPLETE. The bench chain REFUSES to score a run without it.
set -u
cd "$(dirname "$0")"

EPOCHS=${EPOCHS:-100}
OUT=/home/argo/Desktop/Projects/cityscapes/output/models/rfdetr_seg_medium
LOGS=/home/argo/Desktop/Projects/cityscapes/output/logs
mkdir -p "$LOGS"
echo $$ > /tmp/rfdetr_seg.pid

rm -rf "$OUT"                      # fresh run: no mixing of the OOM'd run's state
rm -f "$OUT/TRAINING_COMPLETE"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # the OOM itself recommended this

echo "===================================================================="
echo "  RF-DETR seg-medium  epochs=$EPOCHS  batch=2 x accum=8 (effective 16)  $(date)"
echo "===================================================================="
./.venv-rfdetr/bin/python train_rfdetr.py \
    --variant seg-medium --epochs "$EPOCHS" \
    --batch-size 2 --grad-accum 8 --num-workers 8 \
    2>&1 | tee "$LOGS/rfdetr_seg-medium.log"
rc=${PIPESTATUS[0]}

if [ "$rc" -eq 0 ] && [ -f "$OUT/checkpoint_best_ema.pth" ]; then
  touch "$OUT/TRAINING_COMPLETE"
  echo ">>> RF-DETR seg-medium done $(date)"
else
  echo "!!! RF-DETR seg-medium FAILED (rc=$rc) - NOT marking complete; it must not be benched"
fi
