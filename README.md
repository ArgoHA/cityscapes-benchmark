

# Cityscapes: D-FINE-seg vs YOLO26 vs RF-DETR

A like-for-like benchmark of three detection/segmentation frameworks on Cityscapes, across three
tasks: **detection**, **instance segmentation**, and **semantic segmentation**.

- **[PROTOCOL.md](PROTOCOL.md)** — how every model was trained, exported, timed and scored, plus a
  full list of the asymmetries that remain.
- **[RESULTS.md](RESULTS.md)** / **[RESULTS.csv](RESULTS.csv)** — the numbers.

## The rules this benchmark plays by

1. **Each framework runs its own shipped inference code.** `YOLO(engine).predict()` for
   ultralytics, `TRTInference` + their `PostProcess` for RF-DETR, `TRT_model` for D-FINE. We never
   hand-write a rival's pre/postprocessing — we compare *architecture + the framework it ships in*,
   because that is what a user actually gets.
2. **Accuracy and latency come from the same run**, at each model's **own** val-optimal confidence
   threshold. Reporting F1 at one threshold and latency at another is not a benchmark.
3. **Two latency columns**: the framework's end-to-end call, *and* pure TensorRT engine time. They
   disagree — the rivals' CPU preprocessing dominates end-to-end — and reporting only one of them
   would be misleading.
4. **Sizes are matched on latency, not parameters.** Each rival is the smallest variant with *at
   least* D-FINE's engine budget, so it is never under-resourced (it also carries 2–3× the params).
5. **One scorer for everyone**: D-FINE's `Validator`, original-resolution ground truth. It
   reproduces D-FINE's own `bench.py` output exactly — which is what makes the rival rows
   trustworthy.
6. **Every asymmetry we know about is written down** in [PROTOCOL.md §9](PROTOCOL.md), including
   the ones that don't flatter us.

## Quick start

```bash
python prep_datasets.py                                                  # dataset layouts (symlinks, original resolution)
./run_yolo_training.sh                                                   # train YOLO26
./run_rfdetr_training.sh                                                 # train RF-DETR
uv run --project ../D-FINE-seg python bench_all.py --models dfine yolo rfdetr   # bench: accuracy + latency at the same operating point
uv run --project ../D-FINE-seg python make_results.py                    # regenerate RESULTS.md / RESULTS.csv
```

Requires three isolated venvs (one per framework), each pinned to **TensorRT 10.13.3.9** so all
engines are mutually comparable. Paths are hard-coded at the top of each script — edit them to run
elsewhere.

Hardware for the published numbers: single RTX 5070 Ti (16 GB), TensorRT 10.13.3.9, FP16, batch 1,
idle GPU.
