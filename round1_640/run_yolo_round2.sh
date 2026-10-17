#!/usr/bin/env bash
# YOLO26 round 2: semantic-m (new size, reported alongside semantic-l) + detect-m retrained at batch 10.
#
# detect was originally trained at ultralytics AutoBatch's batch=5 (AutoBatch is broken on this GPU:
# it profiles with empty tensors, over-estimates ~2.4x and under-picks 3-4x). That run is preserved
# at output/models/yolo26m_detect_bs5 (100 ep, best mAP50 0.611 @ ep89) and this one overwrites
# yolo26m_detect. Real measured ceiling for detect-m on this 16 GB card is batch 20, so 10 has 2x
# headroom - a crowded Cityscapes scene cannot OOM the run.
#
# semantic-m: yolo26l-sem was the latency-matched pick (1.62 ms engine vs D-FINE-seg-S 1.52), but m is
# the closer PARAM match. Both get reported.
set -u
cd "$(dirname "$0")/.."  # repo root: python -m round1_640.<script>

EPOCHS=${EPOCHS:-100}
ROOT="${CITYSCAPES_ROOT:-$(cd .. && pwd)}"  # workspace that holds data/ and output/ (paths.py)
LOGS="$ROOT/output/logs"
mkdir -p "$LOGS"
echo $$ > /tmp/yolo_train.pid

RUNS=${RUNS:-"semantic:m:10 detect:m:10"}

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

echo "ALL YOLO ROUND-2 TRAINING DONE $(date)"
