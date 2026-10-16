"""Train YOLO26 on cityscapes (detect / segment / semantic).

Ultralytics defaults everywhere except: dataset, epochs, output dir, seed, and batch=-1
(auto-fit to the GPU, mirroring D-FINE's batch_size=-1). imgsz=640 is already the default,
and letterboxing is left as-is - we compare each framework as it ships.

  .venv-yolo/bin/python -m round1_640.train_yolo --task detect   --size s --epochs 55
  .venv-yolo/bin/python -m round1_640.train_yolo --task segment  --size s --epochs 55
  .venv-yolo/bin/python -m round1_640.train_yolo --task semantic --size s --epochs 75
"""

import argparse

from ultralytics import YOLO

from common.paths import DATA, OUTPUT

OUT = OUTPUT / "models"

# task -> (checkpoint suffix, dataset yaml)
TASKS = {
    "detect": ("", "yolo_det.yaml"),
    "segment": ("-seg", "yolo_seg.yaml"),
    "semantic": ("-sem", "yolo_sem.yaml"),
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=TASKS)
    ap.add_argument("--size", default="s", choices=list("nsmlx"))
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    # Explicit batch, NOT ultralytics' batch=-1 AutoBatch: on this GPU AutoBatch profiles with
    # empty tensors, over-estimates memory ~2.4x and under-picks 3-4x (batch 3 for segment on a
    # free 15.5 GB card). See probe_batch.py / run_yolo_training.sh.
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    suffix, data_yaml = TASKS[args.task]
    weights = f"yolo26{args.size}{suffix}.pt"  # COCO-pretrained
    name = args.name or f"yolo26{args.size}_{args.task}"

    model = YOLO(weights)
    model.train(
        data=str(DATA / data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,
        seed=0,
        project=str(OUT),
        name=name,
        exist_ok=True,
    )
