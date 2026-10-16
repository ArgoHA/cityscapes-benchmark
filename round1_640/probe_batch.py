"""Find the real max batch size for a YOLO26 task on this GPU.

Ultralytics' AutoBatch is badly wrong here: it profiles with EMPTY tensors, so it never sees the
per-instance memory of Cityscapes' crowded scenes, then over-estimates anyway and under-picks
(detect -> batch 5, segment -> batch 3, while actual usage was ~4 GB of 15.5 GB on a free GPU).
Training YOLO at batch 3 while D-FINE's own autobatch picked 10 is not a fair "default" - it's a
broken probe, and small-batch BatchNorm would cost YOLO accuracy.

So: run a few REAL training iterations at each candidate batch and record peak allocated memory.

  .venv-yolo/bin/python -m round1_640.probe_batch --task segment --size m --batches 6 8 10 12
"""

import argparse
import gc
import warnings

warnings.filterwarnings("ignore")

import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from common.paths import DATA  # noqa: E402

TASKS = {
    "detect": ("", "yolo_det.yaml"),
    "segment": ("-seg", "yolo_seg.yaml"),
    "semantic": ("-sem", "yolo_sem.yaml"),
}
TOTAL_GB = torch.cuda.get_device_properties(0).total_memory / 1e9


def probe(task, size, batch, iters=12) -> float | None:
    """Peak GB over a few real train iterations, or None on OOM."""
    suffix, yaml = TASKS[task]
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        YOLO(f"yolo26{size}{suffix}.pt").train(
            data=str(DATA / yaml),
            epochs=1,
            imgsz=640,
            batch=batch,
            device=0,
            workers=4,
            val=False,
            save=False,
            plots=False,
            verbose=False,
            project="/tmp/probe",
            name=f"{task}_{size}_b{batch}",
            exist_ok=True,
            fraction=iters * batch / 2975,  # just enough images for `iters` steps
        )
        return torch.cuda.max_memory_allocated() / 1e9
    except torch.cuda.OutOfMemoryError:
        return None
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            return None
        raise
    finally:
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=TASKS)
    ap.add_argument("--size", required=True)
    ap.add_argument("--batches", type=int, nargs="+", required=True)
    ap.add_argument("--target", type=float, default=0.70, help="fraction of VRAM to aim for")
    args = ap.parse_args()

    print(
        f"\nGPU total {TOTAL_GB:.1f} GB | target {args.target:.0%} = {TOTAL_GB * args.target:.1f} GB"
    )
    ok = []
    for b in args.batches:
        peak = probe(args.task, args.size, b)
        if peak is None:
            print(f"  batch {b:3d}  OOM")
            break
        frac = peak / TOTAL_GB
        fits = frac <= args.target
        print(
            f"  batch {b:3d}  peak {peak:5.2f} GB  ({frac:4.0%}) {'ok' if fits else 'OVER target'}"
        )
        if fits:
            ok.append(b)

    print(f"\n==> largest batch within {args.target:.0%} VRAM: {max(ok) if ok else 'none'}")
