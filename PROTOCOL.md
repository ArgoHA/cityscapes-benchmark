# Benchmark protocol: D-FINE-seg vs YOLO26 vs RF-DETR on Cityscapes

This document is the full experimental protocol behind the numbers in [`RESULTS.md`](RESULTS.md):
how every model was trained, exported, run, timed and scored. It is written to be adversarial
against ourselves — the section [Things that could still make this unfair](#things-that-could-still-make-this-unfair)
lists every remaining asymmetry we know about.

Results: [`RESULTS.md`](RESULTS.md) / [`RESULTS.csv`](RESULTS.csv).

---

## 1. What is being compared, and what is not

**We compare architectures *plus the framework they ship in*.** A user who picks YOLO26 does not
get "the YOLO26 architecture" — they get ultralytics' preprocessing, ultralytics' TensorRT export,
and ultralytics' postprocessing. That whole package is the product, so that whole package is what
we measure.

Concretely, this means the single most important rule of this benchmark:

> **Each framework runs its own shipped inference code. We never hand-write a rival's
> preprocessing, postprocessing, or TensorRT wrapper.**

| framework | inference path used | source |
|---|---|---|
| D-FINE-seg | `src/infer/trt_model.py::TRT_model` (what its own `make bench` uses) | [`infer_dfine.py`](infer_dfine.py) |
| YOLO26 (ultralytics 8.4.92) | `YOLO(engine).predict()` → AutoBackend TensorRT + their predictor | [`infer_yolo.py`](infer_yolo.py) |
| RF-DETR (rfdetr 1.8.3) | `rfdetr.export.benchmark.TRTInference` + their `PostProcess` + their `Normalize` transform | [`infer_rfdetr.py`](infer_rfdetr.py) |

An earlier version of this harness hand-wrote TRT wrappers for the rivals. It was thrown away: a
hand-written preprocessing path is a benchmark of *our* ability to write someone else's
preprocessing, not of their model.

The corollary is that **the end-to-end latency column partly measures preprocessing quality, not
architecture** — so we report engine-only latency next to it (§6).

### Tasks

| task | metric | models |
|---|---|---|
| Detection | F1 @ each model's own optimal confidence | D-FINE-seg · YOLO26 · RF-DETR |
| Instance segmentation | F1 (mask IoU) @ each model's own optimal confidence | D-FINE-seg · YOLO26 · RF-DETR |
| Semantic segmentation | mIoU | D-FINE-seg · YOLO26 (**RF-DETR has no semantic task**) |

---

## 2. Dataset

Cityscapes, official split: **2975 train / 500 val**, images 2048×1024 PNG. All evaluation is on
the 500 official val images, **at their original 2048×1024 resolution** — never at the network's
input resolution.

- **Detection / instance segmentation**: 8 classes — `person, rider, car, truck, bus, train,
  motorcycle, bicycle`.
- **Semantic segmentation**: the 19 standard Cityscapes trainIds (`road … bicycle`).

[`prep_datasets.py`](prep_datasets.py) builds each framework's expected layout from one source
dataset, using **symlinks at original resolution** — no re-encoding, no pre-resizing:

```
data/yolo_det, yolo_seg       images/{train,val} + labels/{train,val}   (ultralytics YOLO layout)
data/yolo_sem                 images/{train,val} + masks/{train,val}    (masks_dir key in the yaml)
data/rfdetr_det, rfdetr_seg   {train,valid}/{images,labels} + data.yaml
```

We deliberately do **not** pre-resize or pre-letterbox the images to normalise geometry across
frameworks. Each framework resizes the 2048×1024 frame however it wants to — that is part of what
we are measuring.

---

## 3. Training

Every model starts from its own framework's **COCO-pretrained** checkpoint and uses that
framework's **default recipe**. We changed only: dataset, epoch budget, output dir, seed, and
(where a framework's autobatch was demonstrably broken) the batch size.

### D-FINE-seg

| | detect | instance seg | semantic seg (S) | semantic seg (m) |
|---|---|---|---|---|
| size | S | S | S | m |
| epochs | 55 | 55 | 75 | 75 |
| batch | 19 (its autobatch) | 10 (its autobatch) | 15 (its autobatch) | 12 (its autobatch) |
| input | 640×640 | 640×640 | 640×640 | 640×640 |
| init | `dfine_s_coco.pt` | `dfine_seg_s_coco.pt` | `dfine_seg_s_coco.pt` | `dfine_seg_m_coco.pt` |
| AMP | bf16 | bf16 | bf16 | bf16 |

### YOLO26 (ultralytics 8.4.92) — [`train_yolo.py`](train_yolo.py)

100 epochs, 640×640, COCO-pretrained init, **batch 10**, everything else ultralytics default
(optimizer `auto` → AdamW, their augmentation, their letterbox, `nbs=64` accumulation).

**Why the batch size is explicit and not `batch=-1`:** ultralytics' AutoBatch is broken on this
GPU. It profiles with *empty* tensors, over-estimates memory ~2.4×, and under-picks by 3–4× — it
chose batch **3** for instance segmentation on a *free* 15.5 GB card, where the real ceiling is 12
(measured by [`probe_batch.py`](probe_batch.py), which runs actual training iterations and reads
`torch.cuda.max_memory_allocated()`). Training a rival at batch 3 because its own probe
misfired is not a "default" — it is a handicap, and 3-sample BatchNorm would have cost YOLO real
accuracy. Measured real ceilings on this 16 GB card: detect 20, segment 12, semantic 12 → we used
**10** everywhere, comfortably inside every ceiling.

**We then verified the batch size does not matter** (see §9, disclosure 2).

### RF-DETR (rfdetr 1.8.3) — [`train_rfdetr.py`](train_rfdetr.py)

100-epoch budget + **rfdetr's own early stopping** (patience 15, `min_delta` 0.001 on EMA mAP —
all their defaults), their own batch/accumulation defaults (batch 4 × grad-accum 4 = effective 16),
and **their own default resolution per size** (medium 576, seg-medium 432). We evaluate their
**best EMA checkpoint**, as rfdetr intends.

**Exception — seg-medium had to be re-run at a smaller micro-batch.** RF-DETR's default
segmentation recipe **runs out of memory on Cityscapes on a 16 GB card**: at ~epoch 22 a crowded
scene made the target-mask tensor in `loss_masks` request 2.70 GiB (`criterion.py:499`) with 1.92
GiB free, and training died. We re-ran it with **`batch_size=2` × `grad_accum_steps=8`**, which
keeps the **effective batch at their default 16** — the optimization recipe is therefore unchanged
(RF-DETR is LayerNorm-based, so there is no batch-statistic dependence); only activation memory
halves. We also set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (recommended by the OOM
message itself) and raised `num_workers` from their default 2 to 8 — pure infrastructure, not a
hyperparameter: their default starves the GPU to ~28% utilization during segmentation training,
because polygon→mask rasterization runs on the CPU. Detect trains fine at their defaults and was
not touched.

**The crashed run left a `checkpoint_best_ema.pth`** (from epoch 16, while the model was still
improving). A checkpoint from a crashed run is **not** a trained model, and benching it would have
silently understated RF-DETR. The training runner now writes a `TRAINING_COMPLETE` marker only on a
clean exit, and the bench harness refuses to score any run that lacks it.

Resolution is how RF-DETR buys its speed, so forcing it to 640 would misrepresent the model. We
match on latency instead (§4).

RF-DETR's default LR schedule is `lr_drop=100` with `epochs=100` — a StepLR whose single drop lands
*after* the last epoch, i.e. **their default recipe is constant-LR throughout**. Early stopping
therefore costs it no end-of-schedule LR decay; there is none to miss.

---

## 4. How model sizes were chosen: latency, not parameters

Parameters are the wrong axis. What a user buys is **speed and accuracy** — and RF-DETR is 30 M+ at
every size, so parameter parity is impossible anyway.

So each rival is the **smallest variant whose engine-only latency is at least D-FINE's**. The rival
is therefore never under-resourced — it gets *more* compute budget than us, and 2–3× the params:

| task | D-FINE-S engine | rival pick | rival engine |
|---|---|---|---|
| detect | 1.38 ms | YOLO26-**m** | 1.61 ms (+17%) |
| segment | 1.91 ms | YOLO26-**m**-seg | 2.09 ms (+9%) |
| semantic | 1.52 ms | YOLO26-**l**-sem | 1.62 ms (+7%) |
| detect | 1.58 ms | RF-DETR **medium** (576) | 1.62 ms |
| segment | 2.09 ms | RF-DETR **seg-medium** (432) | 1.95 ms |

(YOLO26-**s** was tried first and was too weak — detect best mAP50 0.544. It is kept only as a
param-matched reference point, not as the headline rival.)

If D-FINE still wins accuracy against a rival carrying more params *and* a bigger engine budget,
the result is strong.

For semantic segmentation we additionally report **D-FINE-m** (16.0 M), which is the size that
actually matches YOLO26-l/m on latency — reporting only D-FINE-S there was under-matching our own
model, and we say so rather than quietly keeping the flattering row.

---

## 5. Export

**Every engine is TensorRT 10.13.3.9, FP16, batch 1, static shapes (`dynamic=False`)**, built by
its own framework's exporter:

| framework | exporter | flags |
|---|---|---|
| D-FINE | `src/dl/export.py` | `export.half: True`, `dynamic_input: False`, `max_batch_size: 1` → `BuilderFlag.FP16` |
| YOLO26 | ultralytics' own | `YOLO(w).export(format="engine", imgsz=640, half=True)`; `dynamic` left at their default `False` |
| RF-DETR | rfdetr's own ONNX exporter + [`build_engine.py`](build_engine.py) | static `[1,3,576,576]` ONNX → `BuilderFlag.FP16`, 4 GB workspace |

**Why RF-DETR's engine is not built by their `trtexec` helper:** version pinning, not a defect in
their helper. It shells out to whatever `trtexec` is on `PATH` — here the *system* TensorRT 10.11 —
and TRT engines are version-locked, so the 10.13.3.9 runtime in the venv cannot deserialize what it
writes. Since every engine in this benchmark is pinned to 10.13.3.9 for comparability,
[`build_engine.py`](build_engine.py) builds through the TRT Python API at that version, replicating
their exact flags (`--fp16 --memPoolSize=workspace:4096`). Build plumbing only; no model logic is
touched. (Pin the venv's TensorRT to the system `trtexec` version instead and their helper works
unmodified — confirmed in `rfdetr_trt.py`, which produces the same result.)

**Input resolutions are not identical across frameworks** (D-FINE/YOLO 640×640, RF-DETR 576/432).
This is deliberate (§3) and is reported as a column in the results tables.

---

## 6. Latency: two columns, measured separately

### `e2e ms` — the framework's own end-to-end call

This is what a user actually experiences. Protocol (identical for all three, copied from D-FINE's
`src/dl/bench.py` so the numbers are comparable to its own published ones) — see
[`dump.py`](dump.py):

```python
warm = load(images[0])
for _ in range(10):          # 10 warmup calls
    infer(warm)
torch.cuda.synchronize()

for p in images:
    img = load(p)                       # image decode: OUTSIDE the clock
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    native = infer(img)                 # the framework's OWN end-to-end call
    torch.cuda.synchronize()
    lat.append((time.perf_counter() - t0) * 1000)
    out = extract(native)               # .cpu() D2H: OUTSIDE the clock
latency = float(np.mean(lat[1:]))
```

Two things sit **outside** the timed region for every framework, exactly as `bench.py` does it:

1. **Image decode** (`cv2.imread` / `PIL.Image.open`).
2. **Pulling results off the GPU** (`.cpu()`), which `bench.py` performs only after stopping the
   clock. This matters enormously: a `[N,1024,2048]` instance-mask D2H copy costs **~13 ms** on
   Cityscapes and would swamp the number it is supposed to measure — timing it by mistake inflated
   D-FINE-S instance segmentation from 4.4 ms to 17.2 ms. Hence `infer` (timed) and `extract`
   (untimed) are separate callables in [`dump.py`](dump.py).

### `engine ms` — pure TensorRT execute

[`measure_engine.py`](measure_engine.py): same loop, same warmup, same GPU, dummy input, no
H2D/D2H — for every engine. This isolates the architecture and is the number comparable to the
vendors' published TensorRT figures.

### Why both are needed

**The two columns disagree, and that is the point.** Both rivals do heavy CPU preprocessing of the
2048×1024 frame inside their shipped path (ultralytics' letterbox ≈1.7 ms; rfdetr's PIL `Resize`
several ms), while D-FINE does a `cv2.resize` into a pinned buffer. So:

- On **end-to-end**, D-FINE wins partly on preprocessing.
- On **engine-only**, the smaller YOLO26 variants are actually *faster* than D-FINE.

Reporting only e2e would turn the table into a preprocessing contest that D-FINE wins by default.
**Do not present the e2e column as an architectural claim.** Report both.

All latency runs are performed on an **idle GPU** (the harness refuses to bench if any other
process holds the GPU).

---

## 7. Operating point: accuracy and latency must come from the same run

Each framework's confidence scores are calibrated differently (YOLO26 is NMS-free, RF-DETR is
DETR-sigmoid top-k, D-FINE has its own head), so a single fixed threshold flatters some models and
punishes others for reasons that have nothing to do with quality. **Every model is therefore scored
at its own val-optimal confidence.**

But an optimal threshold changes *latency* too — fewer surviving boxes means less NMS work and
fewer masks to upsample to 1024×2048. Reporting F1@0.25 next to latency@0.5 would be measuring two
different operating points, and is not a valid benchmark.

So [`bench_all.py`](bench_all.py) runs **two stages** per model:

```
stage 1   infer at conf=0.05, dump predictions        <- MECHANISM ONLY, never reported
          sweep thresholds over that dump  ->  t* = argmax F1
          (the sweep filters stored predictions, so it can never find an optimum BELOW the
           conf they were dumped at — hence the low floor)

stage 2   infer AGAIN at conf=t*   ->  F1 and latency both come out of THIS run
```

Semantic segmentation is a dense argmax with no confidence threshold, so it is a single pass.

Measured optima: **D-FINE 0.50, YOLO26 0.25** on both box tasks. At a fixed 0.5, YOLO's recall
collapses to 0.475 — reporting that would have overstated our win.

The threshold **sweep** is boxes-only (`ignore_masks=True`), which is D-FINE's own convention for
its `optimal_thresh` column. A mask-based sweep would RLE-decode and mask-IoU-match ~300
predictions/image at 1024×2048 across 500 images × 19 thresholds — hours. The **final** reported
metrics at `t*` are of course full mask-based.

---

## 8. Scoring: one scorer for everyone

[`score.py`](score.py) scores all three frameworks' dumps with **D-FINE's own
`Validator` / `SemSegValidator`**, against ground truth built by its `BenchLoader`, at original
2048×1024 resolution. One scorer, one GT pipeline, one metric definition.

**The validation that makes the rival rows trustworthy:** re-scoring D-FINE's own predictions
through this harness reproduces its stored `bench.py` output *exactly*:

| | this harness | `bench_metrics.csv` |
|---|---|---|
| detect F1 / P / R / IoU | 0.703 / 0.817 / 0.617 / 0.446 | 0.703 / 0.817 / 0.617 / 0.446 |
| detect TP / FP / FN | 6285 / 1411 / 3895 | 6285 / 1411 / 3895 |
| instance seg F1 | 0.661 | 0.661 |
| sem_seg mIoU / pixel acc | 0.728 / 0.950 | 0.728 / 0.950 |

If the harness reproduces the reference implementation bit-for-bit on the model we *do* control, it
is not silently mangling the models we don't.

---

## 9. Things that could still make this unfair

Every asymmetry we are aware of, stated plainly.

1. **Sizes are latency-matched, not param-matched.** Each rival carries 2–3× the params and a
   *larger* engine budget than D-FINE (§4). This is generous to the rival, by design — but it does
   mean "D-FINE beats a model with 2.2× the params" is a latency-normalised claim, not a
   parameter-normalised one.

2. **YOLO26 detect batch size — resolved empirically.** The first detect run used ultralytics'
   AutoBatch pick (batch 5). Because AutoBatch is demonstrably broken here, we retrained at batch
   10 to remove the objection. Result: batch 10 was **worse** on ultralytics' own val mAP50 (0.593
   vs 0.611) and **identical** on our F1 (0.689 vs 0.690 — it merely trades precision for recall).
   Cause: ultralytics accumulates to a fixed `nbs=64`, so doubling the batch halves the optimizer
   steps per epoch. Both runs are in the results table; we report YOLO's better one. **The batch
   size is not what decides this benchmark.**

3. **Epoch budgets differ**: rivals 100, D-FINE 55 (det/seg) / 75 (sem_seg). This favours the
   rivals, not us — and D-FINE was already overfitting at 55.

4. **RF-DETR stopped itself early** (~epoch 32 for medium, peaking at 17) via its own early-stopping
   defaults. As shown in §3 this costs it no LR decay, and its best EMA checkpoint is what we
   evaluate — but it *is* fewer epochs than YOLO got.

4b. **RF-DETR seg-medium's micro-batch was halved** (2 × accum 8 instead of 4 × accum 4) because its
   default recipe OOMs on this dataset/GPU (§3). The **effective batch is unchanged at 16**, so we
   consider this memory plumbing rather than a recipe change — but it is a deviation from the
   framework's defaults, and it is the only one in this benchmark. `num_workers` 2 → 8 is pure I/O
   and cannot affect accuracy.

5. **Input resolution differs** (640 / 640 / 576 / 432). Deliberate — resolution is a design choice
   each framework makes to trade accuracy for speed, and latency is the axis we match on. Reported
   as a column.

6. **The e2e latency column is not an architectural claim.** It includes each framework's CPU
   preprocessing. On engine-only latency the smaller YOLO26 variants beat D-FINE. Both columns are
   always reported together (§6).

7. **RF-DETR has no semantic segmentation task**, so it is simply absent from that table.

8. **Two upstream bugs in rfdetr 1.8.3** had to be worked around. Both are plumbing —
   **no model logic and no pre/postprocessing was modified**:

   a. `export/benchmark.py` does `import _tensorrt as trt` (a typo — the module is `tensorrt`), so
      `TRTInference` *always* raises "TensorRT is not installed", even where TensorRT is installed.
      Their only shipped engine runner therefore cannot run at all. We alias the module.

   b. **Silent-garbage bug — the important one.** `rfdetr.datasets.transforms.Normalize` returns a
      **non-contiguous** tensor, and `TRTInference` passes `.data_ptr()` straight to TensorRT,
      which assumes contiguous NCHW. Feeding their own transform's output to their own TRT wrapper
      therefore makes the engine read **scrambled memory**: logits come out all-negative (max
      −2.1 instead of +3.0), every score collapses below 0.11, and F1 reads **~0.00** — with no
      error raised, and boxes that still *look* plausible. Fix: `.contiguous()` before the call
      (same values, correct layout). **Anyone benchmarking RF-DETR on TensorRT via their shipped
      path is exposed to this**: their own `benchmark.infer_transforms()` produces a non-contiguous
      tensor too (and hardcodes `Resize((640, 640))`, which does not match the input binding of any
      model that is not 640). We caught it by bisecting: torch `predict()` ✓ → ONNX on CPU ✓ →
      engine driven by our own runner ✓ → so only the hand-off was left.

   Issue (b) is why an early version of this table showed RF-DETR at F1 0.001. That number was a
   bug in *our* harness, not a property of RF-DETR, and publishing it would have been a false
   result in our own favour.

   **Verified independently (2026-07-14).** RF-DETR was re-run from scratch through its *documented*
   TensorRT path — `model.export()` → their `export._tensorrt.trtexec()` helper → their
   `TRTInference` → their `PostProcess`, with preprocessing copied from `RFDETR.predict()` — in a
   venv whose TensorRT is pinned to the same version as the `trtexec` binary on `PATH`. Only the
   two bugs above had to be worked around. That run reproduces this run's numbers (17.7 ms e2e vs
   16.4 ms here, the difference being `predict()`'s full-resolution H2D) and, crucially, reproduces
   RF-DETR's **own** training-time validation metrics on the same 500 images — mask mAP@50 0.583 vs
   0.578 logged, mask mAP@50-95 0.334 vs 0.329, box mAP@50 0.611 vs 0.612 — which is what proves the
   engine we benched is not silently degraded. Script: `rfdetr_trt.py`.

---

## 10. Reproducing

Three isolated venvs (each framework pinned to **TensorRT 10.13.3.9**, so all engines are
mutually comparable), plus `.venv-rfdetr-trtexec` (TRT **10.11**, matching the system `trtexec`)
used only by the `rfdetr_trt.py` cross-check in step 5:

```bash
# 1. dataset layouts for each framework (symlinks, original resolution)
python prep_datasets.py

# 2. train
./run_yolo_training.sh                    # YOLO26: detect, segment, semantic
./run_rfdetr_training.sh                  # RF-DETR: medium (detect), seg-medium (segment)
# D-FINE is trained in its own repo:  python -m src.dl.train exp_name=... task=...

# 3. bench everything — accuracy AND latency at each model's own optimal operating point
python bench_all.py --models dfine yolo rfdetr

# 4. regenerate RESULTS.md / RESULTS.csv from the bench CSVs (nothing hand-typed)
python make_results.py

# 5. (independent cross-check) RF-DETR through its OWN documented TensorRT path — their trtexec
#    helper, their TRTInference, their PostProcess. Needs the TRT-10.11 venv, matching the system
#    trtexec binary; verifies the engine against RF-DETR's own training-time val metrics (§9.8).
.venv-rfdetr-trtexec/bin/python rfdetr_trt.py
```

`bench_all.py` orchestrates: it calls each framework's inference script **in that framework's own
venv**, runs the two-stage threshold protocol (§7), measures engine-only latency in the same
idle-GPU session, and scores every dump through the single shared scorer (§8).

### Scripts

| file | role |
|---|---|
| [`prep_datasets.py`](prep_datasets.py) | build each framework's dataset layout (symlinks, original res) |
| [`probe_batch.py`](probe_batch.py) | measure the *real* max batch by running actual training iterations |
| [`train_yolo.py`](train_yolo.py) / [`run_yolo_training.sh`](run_yolo_training.sh) | YOLO26 training |
| [`run_yolo_round2.sh`](run_yolo_round2.sh) | the batch-10 detect retrain + semantic-m run behind disclosure 2 (§9) |
| [`train_rfdetr.py`](train_rfdetr.py) / [`run_rfdetr_training.sh`](run_rfdetr_training.sh) | RF-DETR training |
| [`dump.py`](dump.py) | shared timing protocol + prediction dump format (§6) |
| [`infer_dfine.py`](infer_dfine.py) / [`infer_yolo.py`](infer_yolo.py) / [`infer_rfdetr.py`](infer_rfdetr.py) | run each framework's **own** shipped inference |
| [`build_engine.py`](build_engine.py) | ONNX → TRT FP16 at the venv's TRT version (RF-DETR) |
| [`measure_engine.py`](measure_engine.py) | pure TRT execute latency, uniform across all engines |
| [`score.py`](score.py) | score every dump with D-FINE's `Validator`, original-res GT (§8) |
| [`bench_all.py`](bench_all.py) | **entrypoint** — the two-stage protocol end to end |
| [`make_results.py`](make_results.py) | generate `RESULTS.md` / `RESULTS.csv` from the bench CSVs |

> **Note on paths:** the scripts currently hard-code absolute paths to the dataset and model
> directories at the top of each file (`ROOT`, `DATA`, `M`, …). Edit those to run elsewhere.

### Hardware

Single NVIDIA RTX 5070 Ti (16 GB), TensorRT 10.13.3.9, FP16, batch 1. All latency measurements on
an idle GPU.
