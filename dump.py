"""Shared timing protocol + prediction dump format for the 3-framework comparison.

Timing is copied from D-FINE's src/dl/bench.py so every model is measured identically:
  10 warmup calls, then per image: cuda-sync -> perf_counter -> <framework's own end-to-end
  inference call> -> cuda-sync. Reported latency = mean(latency[1:]).

Two things sit OUTSIDE the timed region for EVERY framework, matching bench.py exactly:
  - image decode (cv2.imread / Image.open)
  - pulling results off the GPU (`.cpu()`), which bench.py does only after stopping the clock.
    This matters: a [N,1024,2048] instance-mask D2H copy costs ~13 ms on Cityscapes and would
    otherwise swamp the number it's supposed to measure (it inflated D-FINE-S segment from
    4.4 -> 17.2 ms when timed by mistake). So `infer` returns the framework's native (GPU)
    result and `extract` converts it afterwards.

Predictions are dumped so a single scorer (D-FINE's Validator, original-resolution GT) can score
all three frameworks from the same code, and so thresholds can be re-swept without re-inferring.
"""

import pickle
import time
from pathlib import Path
from typing import Callable, List

import numpy as np
import torch
from tqdm import tqdm


def rle_encode(masks: np.ndarray) -> List[dict]:
    """pycocotools-compatible RLE (same backend D-FINE's bench uses to keep memory bounded)."""
    from faster_coco_eval.core import mask as mask_utils

    if masks is None or len(masks) == 0:
        return []
    m = np.asfortranarray(masks.astype(np.uint8).transpose(1, 2, 0))
    return mask_utils.encode(m)


def run(
    name: str,
    task: str,
    infer: Callable,  # (decoded_image) -> framework-native result  [TIMED]
    extract: Callable,  # (native result) -> dict(boxes,scores,labels[,masks]) | dict(sem_seg)
    images: List[Path],
    load: Callable,  # (Path) -> decoded image  [NOT timed]
    out_dir: Path,
    extra: dict | None = None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    sem_dir = out_dir / "sem_seg"
    if task == "sem_seg":
        sem_dir.mkdir(exist_ok=True)

    warm = load(images[0])
    for _ in range(10):
        infer(warm)
    torch.cuda.synchronize()

    lat, preds = [], []
    for p in tqdm(images, desc=name):
        img = load(p)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        native = infer(img)
        torch.cuda.synchronize()
        lat.append((time.perf_counter() - t0) * 1000)

        out = extract(native)  # D2H copies happen off the clock, as in bench.py

        if task == "sem_seg":
            import cv2

            cv2.imwrite(str(sem_dir / f"{p.stem}.png"), out["sem_seg"])
            preds.append({"file": p.name})
        else:
            rec = {
                "file": p.name,
                "boxes": out["boxes"].astype(np.float32),
                "scores": out["scores"].astype(np.float32),
                "labels": out["labels"].astype(np.int64),
            }
            if "masks" in out:
                m = out["masks"]
                rec["masks_rle"] = rle_encode(m)
                rec["masks_size"] = tuple(m.shape[-2:]) if len(m) else (0, 0)
            preds.append(rec)

    latency = float(np.mean(lat[1:]))
    blob = {
        "model": name,
        "task": task,
        "latency_ms": latency,
        "n_images": len(images),
        "preds": preds,
        **(extra or {}),
    }
    with open(out_dir / "preds.pkl", "wb") as f:
        pickle.dump(blob, f, protocol=4)

    print(f"\n{name}: latency {latency:.2f} ms/img over {len(images)} images -> {out_dir}")
    return blob
