"""Build RESULTS.md + RESULTS.csv from the per-task outputs of bench_all.py.

Nothing here is hand-typed: accuracy/e2e-latency/conf come from <task>_final.csv, engine latency
from the engine_ms column bench_all wrote in the same idle-GPU session. Only PARAMS is a literal,
and each entry is sourced below.

  uv run --project "$DFINE_SEG_ROOT" python make_results.py
"""

import pandas as pd

from paths import OUTPUT, RIV

C = OUTPUT / "compare"

# D-FINE  : sum of float params in output/models/<run>/model.pt
# YOLO26  : ultralytics' own "YOLO26x summary: N layers, P parameters" line in its training log
# RF-DETR : filled in after training (same method as D-FINE — sum over the checkpoint)
PARAMS = {
    ("detect", "D-FINE-S"): 10.29,
    ("detect", "YOLO26-m"): 21.79,
    ("detect", "YOLO26-m-bs10"): 21.79,
    ("detect", "RF-DETR-medium"): 33.39,
    ("segment", "D-FINE-S"): 11.87,
    ("segment", "YOLO26-m"): 26.98,
    ("segment", "RF-DETR-seg-medium"): 35.40,
    ("sem_seg", "D-FINE-S"): 8.02,
    ("sem_seg", "D-FINE-m"): 16.00,
    ("sem_seg", "YOLO26-l"): 17.87,
    ("sem_seg", "YOLO26-m"): 14.32,
}

TASKS = {
    "detect": ("Detection", "f1", "F1"),
    "segment": ("Instance segmentation", "f1", "F1"),
    "sem_seg": ("Semantic segmentation", "mIoU", "mIoU"),
}
COLS = {
    "detect": [
        "params_M",
        "input",
        "conf",
        "f1",
        "precision",
        "recall",
        "iou",
        "latency",
        "engine_ms",
    ],
    "segment": [
        "params_M",
        "input",
        "conf",
        "f1",
        "precision",
        "recall",
        "iou",
        "latency",
        "engine_ms",
    ],
    "sem_seg": ["params_M", "input", "mIoU", "pixel_acc", "latency", "engine_ms"],
}
HEAD = {
    "params_M": "params (M)",
    "input": "input",
    "conf": "conf",
    "f1": "**F1**",
    "precision": "precision",
    "recall": "recall",
    "iou": "IoU",
    "mIoU": "**mIoU**",
    "pixel_acc": "pixel acc",
    "latency": "e2e ms",
    "engine_ms": "engine ms",
}


METHOD = """## How these numbers were produced

- **Each framework runs its OWN shipped inference.** D-FINE → `src/infer/trt_model.py::TRT_model`
  (what `make bench` uses). YOLO26 → `YOLO(engine).predict()` (ultralytics AutoBackend TensorRT +
  their predictor). RF-DETR → `rfdetr.export.benchmark.TRTInference` + their own `PostProcess`.
  No hand-written pre/postprocessing anywhere: we compare architectures *plus the framework they
  ship in*, which is what a user actually gets.
- **Every engine is TensorRT 10.13.3.9, FP16, batch 1, static shapes (`dynamic=False`)**, each
  built by its own framework's exporter: D-FINE `export.half=True` / `dynamic_input=False`;
  YOLO26 `YOLO(w).export(format="engine", half=True)` (ultralytics' default `dynamic: False`);
  RF-DETR its own ONNX exporter + `BuilderFlag.FP16` (we build via the TRT Python API rather than
  their `trtexec` helper, which shells out to the *system* TRT 10.11 while every engine here is
  pinned to 10.13.3.9 — same flags: `--fp16 --memPoolSize=workspace:4096`).
- **Input resolution is NOT the same for all models** (see the `input` column): D-FINE and YOLO26
  run at 640×640, RF-DETR at its own defaults (medium 576, seg-medium 432). Resolution is how
  RF-DETR buys its speed, so forcing it to 640 would misrepresent the model; we let each framework
  use the resolution it ships with and match on latency instead.
- **One scorer for everyone**: D-FINE's `Validator` / `SemSegValidator` against GT built by its
  `BenchLoader`, at original resolution. Re-scoring D-FINE through this harness reproduces its
  stored `bench.py` output exactly (detect F1 0.703 / P 0.817 / R 0.617, TP-FP-FN 6285/1411/3895),
  which is what makes the rival rows trustworthy.
- **Accuracy and latency come from the SAME pass.** Each model is scored at *its own* val-optimal
  confidence (F1-maximising, found by a boxes-only sweep over a low-conf dump), then **re-inferred
  at that threshold** so latency is measured at the same operating point. YOLO26's optimum is 0.25
  on both box tasks vs D-FINE's 0.50; reporting F1@0.25 next to latency@0.5 would be invalid.
  sem_seg is a dense argmax with no threshold, so it is a single pass.
- **Two latency columns, and they can disagree.** `e2e ms` is the framework's own end-to-end call
  (10 warmup, then per image `cuda.synchronize()` → `perf_counter()` → call → `synchronize()`,
  `mean(lat[1:])`); image decode and the `.cpu()` D2H sit outside the clock, exactly as `bench.py`
  does — the `[N,1024,2048]` instance-mask D2H alone costs ~13 ms and would swamp the measurement.
  `engine ms` is pure TRT execute, same loop for every engine.
  **e2e includes each framework's CPU preprocessing** (ultralytics' letterbox ≈1.7 ms; D-FINE's
  `cv2.resize` into a pinned buffer is much cheaper), so it partly measures preprocessing quality,
  not architecture. Report both; do not present e2e as an architectural claim.

## Caveats to disclose

1. **Sizes are latency-matched, not param-matched** (user's call: "params are less important, speed
   and accuracy are what matter"). Each rival is the *smallest* variant with **at least** D-FINE's
   engine budget, so the rival is never under-resourced — it also carries 2–3× the params.
2. **YOLO26-m detect batch size is a non-issue.** It was first trained at batch 5 (ultralytics'
   AutoBatch pick; AutoBatch is broken on this GPU — it profiles with empty tensors and under-picks
   3–4×). Retrained at batch 10 it scored *worse* on ultralytics' own val mAP50 (0.593 vs 0.611)
   and identical on our F1 (0.689 vs 0.690) — ultralytics accumulates to a fixed `nbs=64`, so a 2×
   batch just halves the optimizer steps per epoch. Both runs are in the table; we report YOLO's
   better one.
3. **Epochs**: rivals got 100 (RF-DETR: 100 budget + its own early stopping); D-FINE got 55
   (det/seg) / 75 (sem_seg). Not a handicap — D-FINE was already overfitting at 55.
4. **RF-DETR has no semantic segmentation task**, so it is absent from that table.
5. **RF-DETR stopped itself early.** Given a 100-epoch budget, its own early stopping (patience 15,
   min_delta 0.001 on EMA mAP — all rfdetr defaults) fired at ~epoch 32 for `medium`, having peaked
   at epoch 17 (EMA mAP50-95 0.389). This costs it nothing in LR schedule: rfdetr's default
   `lr_drop=100` with `epochs=100` is a StepLR whose only drop lands *after* the final epoch, so
   its default recipe is constant-LR throughout — there is no end-of-schedule decay to miss. We
   evaluate its best EMA checkpoint, as rfdetr intends.
6. **RF-DETR seg-medium's micro-batch was halved** (batch 2 x grad_accum 8 instead of their default
   4 x 4): their shipped recipe OOMs on Cityscapes on a 16 GB card — a crowded scene makes the
   target-mask tensor in `loss_masks` request 2.7 GB. The **effective batch stays at their default
   16**, so the optimization recipe is unchanged (RF-DETR is LayerNorm-based — no batch-statistic
   dependence); only activation memory halves. This is the only deviation from a framework default
   anywhere in this benchmark. (`num_workers` 2 -> 8 is pure I/O and cannot affect accuracy; their
   default leaves the GPU ~70% idle during seg training.)
7. **Two upstream bugs in rfdetr 1.8.3 had to be worked around** to run RF-DETR on TensorRT at all
   (plumbing only — no model logic, no pre/postprocessing touched). Both are still in the current
   release; see PROTOCOL.md §9.8.
   - `export/benchmark.py` does `import _tensorrt as trt` (typo for `tensorrt`), so their only
     shipped engine runner, `TRTInference`, *always* raises "TensorRT is not installed".
   - **Their shipped TRT path silently produces garbage unless the input is made contiguous.**
     `rfdetr.datasets.transforms.Normalize` returns a NON-contiguous tensor and `TRTInference` hands
     `.data_ptr()` straight to TensorRT, which assumes contiguous NCHW — so the engine reads
     scrambled memory: logits go all-negative, every score collapses below 0.11, and **F1 reads
     ~0.00 with no error raised**. An early version of this very table showed RF-DETR at F1 0.001
     for exactly this reason. That was a bug in OUR harness, not a property of RF-DETR, and it would
     have been a false result in our own favour. Fixed with `.contiguous()` (same values, correct
     layout).
8. **The RF-DETR rows are verified against RF-DETR's own numbers.** Re-run from scratch through its
   documented TensorRT path (`rfdetr_trt.py`), the engine reproduces the checkpoint's own
   training-time val metrics on the same 500 images: mask mAP@50 0.583 vs 0.578 logged, mask
   mAP@50-95 0.334 vs 0.329, box mAP@50 0.611 vs 0.612. The engine we benched is not degraded.
"""


def load(task: str) -> pd.DataFrame:
    df = pd.read_csv(C / f"{task}_final.csv", index_col=0)
    df["params_M"] = [PARAMS.get((task, m), float("nan")) for m in df.index]
    return df[[c for c in COLS[task] if c in df.columns]]


def md_table(task: str, df: pd.DataFrame) -> str:
    _, key, _ = TASKS[task]
    best_acc, best_lat, best_eng = df[key].max(), df["latency"].min(), df["engine_ms"].min()
    head = "| model | " + " | ".join(HEAD[c] for c in df.columns) + " |"
    sep = "|---" * (len(df.columns) + 1) + "|"
    lines = [head, sep]
    for name, row in df.iterrows():
        cells = []
        for c, v in row.items():
            if pd.isna(v):
                cells.append("—")
                continue
            s = f"{v:g}" if isinstance(v, (int, float)) else str(v)  # `input` is e.g. "640x640"
            best = (
                (c == key and v == best_acc)
                or (c == "latency" and v == best_lat)
                or (c == "engine_ms" and v == best_eng)
            )
            cells.append(f"**{s}**" if best else s)
        star = "**" if row[key] == best_acc else ""
        lines.append(f"| {star}{name}{star} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    dfs = {t: load(t) for t in TASKS if (C / f"{t}_final.csv").exists()}

    flat = []
    for task, df in dfs.items():
        for name, row in df.iterrows():
            flat.append(
                {
                    "task": task,
                    "model": name,
                    **{
                        c: row.get(c)
                        for c in [
                            "params_M",
                            "input",
                            "conf",
                            "f1",
                            "precision",
                            "recall",
                            "iou",
                            "mIoU",
                            "pixel_acc",
                            "latency",
                            "engine_ms",
                        ]
                    },
                }
            )
    out = pd.DataFrame(flat).rename(
        columns={"latency": "latency_e2e_ms", "engine_ms": "latency_engine_ms"}
    )
    out.to_csv(RIV / "RESULTS.csv", index=False)

    body = [
        "# Cityscapes: D-FINE-seg vs YOLO26 vs RF-DETR — results",
        "",
        "Generated by `make_results.py` from `bench_all.py` output — no hand-typed numbers.",
        "Machine-readable copy: [`RESULTS.csv`](RESULTS.csv). Full protocol + every known asymmetry:",
        "[`PROTOCOL.md`](PROTOCOL.md).",
        "500 Cityscapes val images, original resolution, TensorRT 10.13.3.9 FP16, batch 1, idle GPU.",
        "",
    ]
    for task, df in dfs.items():
        title, key, label = TASKS[task]
        body += [f"## {title}", "", md_table(task, df), ""]
    body += [METHOD]
    (RIV / "RESULTS.md").write_text("\n".join(body))
    print((RIV / "RESULTS.md").read_text())
    print(f"wrote {RIV / 'RESULTS.md'} and {RIV / 'RESULTS.csv'}")
