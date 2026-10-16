#!/usr/bin/env bash
# YOLO26 on Cityscapes: detect -> segment -> semantic, in sequence.
#
# Sizes are LATENCY-matched to D-FINE-S, not param-matched. Params are the wrong axis here
# (RF-DETR is 30M+ at every size, so param parity is impossible anyway); what a user buys is
# speed and accuracy. Engine-only TRT fp16 @640, measured on an idle GPU:
#
#     task      D-FINE-S   YOLO26 pick        note
#     detect      1.38     m  1.61 ms         +17% budget
#     segment     1.91     m  2.09 ms         +9%
#     semantic    1.52     l  1.62 ms         +7%
#
# i.e. each YOLO is the SMALLEST variant that gets AT LEAST D-FINE's engine budget, so YOLO is
# never handicapped (it also carries 2-3x the params). If D-FINE still wins accuracy, the result
# holds. YOLO26-s was tried first and is too weak; its detect run (best mAP50 0.544 @ epoch 83)
# is kept as a param-matched reference point.
#
# 100 epochs; everything else ultralytics default (imgsz=640, letterbox, batch=-1, COCO init).
#
# BATCH SIZE IS SET EXPLICITLY - ultralytics' AutoBatch is broken on this GPU. It profiles with
# empty tensors, over-estimates by ~2.4x, and under-picks by 3-4x: it chose batch 5 for detect and
# batch 3 (!) for segment while actually using ~4 GB of a free 15.5 GB card, and D-FINE's own
# autobatch picked 19 / 10 for the same tasks. Training YOLO at batch 3 is not a fair "default" -
# it's a broken probe, and 3-sample BatchNorm would cost YOLO real accuracy. probe_batch.py runs
# REAL training iterations and reads peak allocated memory; measured ceilings on this 16 GB card:
#     detect   m : autobatch 5  -> real max 20 @69% VRAM
#     segment  m : autobatch 3  -> real max 12 @56% VRAM (16 OOMs)
#     semantic l : ...          -> real max 12 @64% VRAM
# We use batch=10 everywhere: comfortably inside every ceiling so a crowded Cityscapes scene can't
# OOM an unattended run, same batch D-FINE's own autobatch chose for segment, and still 2-3x what
# AutoBatch picked. Optimizer batch is unaffected either way (ultralytics accumulates to nbs=64).
set -u
cd "$(dirname "$0")/.."  # repo root: python -m round1_640.<script>

EPOCHS=${EPOCHS:-100}
ROOT="${CITYSCAPES_ROOT:-$(cd .. && pwd)}"  # workspace that holds data/ and output/ (paths.py)
LOGS="$ROOT/output/logs"
mkdir -p "$LOGS"
echo $$ > /tmp/yolo_train.pid   # chain waiter polls this; pgrep matched a transient wrapper before

# task:size:batch — sizes latency-matched to D-FINE-S, batches measured (see above).
# detect is NOT re-run: yolo26m_detect already trained to convergence (best mAP50 0.606 @ ep99,
# flat over epochs 60-100) but at AutoBatch's batch=5. Kept by decision; the batch-5-vs-19
# asymmetry MUST be disclosed in the writeup - detect is the closest row (0.606 vs D-FINE 0.613).
RUNS=${RUNS:-"segment:m:10 semantic:l:10"}

for spec in $RUNS; do
  IFS=: read -r task size batch <<< "$spec"
  echo "===================================================================="
  echo "  YOLO26-$size  $task   epochs=$EPOCHS  batch=$batch   $(date)"
  echo "===================================================================="
  ./.venv-yolo/bin/python -m round1_640.train_yolo \
      --task "$task" --size "$size" --epochs "$EPOCHS" --batch "$batch" \
      2>&1 | tee "$LOGS/yolo26${size}_${task}.log"
  rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then
    echo "!!! YOLO26-$size $task FAILED (rc=$rc) - continuing to next task"
  else
    echo ">>> YOLO26-$size $task done  $(date)"
  fi
done

echo "ALL YOLO TRAINING DONE $(date)"
