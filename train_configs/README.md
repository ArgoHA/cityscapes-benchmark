# D-FINE-seg training configs: Cityscapes semantic segmentation at native 1024×2048

These are the exact configs behind the five D-FINE-seg rows of the native sem-seg benchmark
([`../README.md`](../README.md)). Each file is the `config.yaml` that D-FINE-seg saved into its run
folder, with two changes: `train.root` is a placeholder, and the Hydra service block from
D-FINE-seg's root `config.yaml` is appended (saved run configs drop it, and without it Hydra
changes the working directory).

| file | size | run folder (original) | init |
|---|---|---|---|
| [`dfine_n.yaml`](dfine_n.yaml) | N | `native_n_2026-10-15` | `dfine_n_coco.pt` (COCO detection) |
| [`dfine_s.yaml`](dfine_s.yaml) | S | `native_from_obj_s_2026-10-13` | `dfine_s_obj2coco.pt` (Objects365 → COCO detection) |
| [`dfine_m.yaml`](dfine_m.yaml) | M | `native_m_2026-10-15` | `dfine_m_obj2coco.pt` |
| [`dfine_l.yaml`](dfine_l.yaml) | L | `native_l_2026-10-14` | `dfine_l_obj2coco.pt` |
| [`dfine_x.yaml`](dfine_x.yaml) | X | `native_x_2026-10-14` | `dfine_x_obj2coco.pt` |

S, M, L and X start from the Objects365 → COCO detection checkpoints. N starts from COCO
detection, because no `dfine_n_obj2coco.pt` exists. YOLO26-sem starts from COCO instance-seg
weights and does not use Objects365, so this is an init asymmetry in D-FINE-seg's favour. The
mask decoder and the sem-seg head are not in either detection checkpoint and start from random
init. On S the init was compared directly (same recipe, one seed each, best-epoch val mIoU):
Objects365 → COCO detection 81.05, COCO detection 80.88 (a 100-epoch run, the others 70),
COCO instance seg 80.37.

## Code version

All five runs used [D-FINE-seg](https://github.com/ArgoHA/D-FINE-seg) commit
[`d99b0c8`](https://github.com/ArgoHA/D-FINE-seg/commit/d99b0c88a2949c4fdbe9dcabdb5b9ddf4ff78a34)
("fix dynamic input size for onnx and openvino", one commit after `v0.5.1`), with no code
changes: the only uncommitted difference in `dfine_seg/` was an import reorder. Only the run
config differs from the defaults.

## Recipe: what differs from D-FINE-seg's defaults

Compared with the root `config.yaml` at `d99b0c8`. Every key not listed is the default.

Same for all five sizes:

| key | default | these runs |
|---|---|---|
| `task` | `detect` | `sem_seg` |
| `train.img_size` | `[640, 640]` | `[1024, 2048]` (native, no resize: `keep_ratio: False` is the default) |
| `train.data_path` | `${train.root}/data/dataset` | `${train.root}/data/dataset_sem` |
| `train.label_to_name` | 8 Cityscapes thing classes | the 19 Cityscapes trainIds |
| `export.formats` | `[onnx, tensorrt, openvino, coreml]` | `[tensorrt]` |
| `bench.formats` | `[torch, tensorrt, onnx, openvino, coreml]` | `[torch, tensorrt]` |

Per size:

| key | default | N | S | M | L | X |
|---|---|---|---|---|---|---|
| `model_name` | `s` | `n` | `s` | `m` | `l` | `x` |
| `exp_name` | `det_s` | `native_n` | `native_from_obj_s` | `native_m` | `native_l` | `native_x` |
| `train.batch_size` | `-1` (autobatch) | 5 | 3 | 3 | 2 | 2 |
| `train.epochs` | 55 | 75 | 70 | 70 | 65 | 70 |
| `train.pretrained_dataset` | `coco` | `coco` | `coco` ¹ | `obj2coco` | `obj2coco` | `obj2coco` |

¹ S sets `train.pretrained_model_path: pretrained/dfine_s_obj2coco.pt` directly, so its init is
Objects365 → COCO like M/L/X.

The rest is the default recipe: bf16 AMP, Muon for the encoder/decoder attention and MLP
matrices with Adan for the other parameters, the per-size learning rates of `train.lrs`, EMA
0.9998, mosaic (the sem-seg default probability 0.5, switched off for the last 5 epochs),
`left_right_flip: 0.3`, seed 42. `train.use_wandb` is `true`; set it to `false` without a W&B
login.

## Training outcome

Read from each run's `train_log.txt`. Val numbers here are D-FINE-seg's own torch evaluation
during training; the benchmark table uses the TensorRT fp16 engine of the best epoch
(`model.pt`), scored by the harness.

| size | init | batch | epochs set / run | best epoch | val mIoU / pixel acc, best epoch | val mIoU / pixel acc, last epoch | train time, h |
|---|---|---:|---|---:|---|---|---:|
| N | `dfine_n_coco.pt` | 5 | 75 / 75 | 57 | 78.00 / 96.02 | 77.51 / 95.97 | 5.56 |
| S | `dfine_s_obj2coco.pt` | 3 | 70 / 70 | 42 | 81.05 / 96.47 | 80.43 / 96.41 | 6.36 |
| M | `dfine_m_obj2coco.pt` | 3 | 70 / 66 (stopped) | 27 | 81.45 / 96.56 | 80.52 / 96.55 | 7.32 |
| L | `dfine_l_obj2coco.pt` | 2 | 65 / 65 | 38 | 83.88 / 96.95 | 83.62 / 96.91 | 8.48 |
| X | `dfine_x_obj2coco.pt` | 2 | 70 / 62 (stopped) | 42 | 84.80 / 97.10 | 84.57 / 97.03 | 11.28 |

`model.pt` is the best-val epoch, selected on the same val set the benchmark reports. That
selection bias is why the last-epoch numbers are listed too. M and X were stopped by hand at
epoch 66 and 62 of 70 because they were overfitting: no val gain after epochs 27 and 42.
Training ran on one RTX 5070 Ti (16 GB).

## Reproduce

Workspace layout, as in the benchmark ([`../paths.py`](../paths.py)): `<root>/data/dataset_sem`,
`<root>/output/models`, with this repo at `<root>/cityscapes-benchmark`.

```bash
# 1. D-FINE-seg at the commit the runs used
git clone https://github.com/ArgoHA/D-FINE-seg && cd D-FINE-seg && git checkout d99b0c8
uv sync                                    # includes TensorRT 10.13.3.9
export DFINE_SEG_ROOT=$PWD

# 2. dataset: the official Cityscapes zips -> <root>/data/dataset_sem (19 trainIds, 2975/500)
cd <root>/cityscapes-benchmark
uv run --project "$DFINE_SEG_ROOT" python prep_datasets.py --only dataset_sem \
    --raw /path/to/cityscapes_raw          # holds the unzipped leftImg8bit/ and gtFine/

# 3. train: one config dir per size; set train.root in the copy (or pass train.root=... below)
mkdir -p ~/dfine_cfg_l && cp train_configs/dfine_l.yaml ~/dfine_cfg_l/config.yaml
cd "$DFINE_SEG_ROOT"
DFINE_SEG_CONFIG_DIR=~/dfine_cfg_l uv run dfine train train.root=<root>
#    -> <root>/output/models/native_l_<date>/model.pt  (pretrained weights download from HF)

# 4. export: TensorRT fp16, static 1024x2048, batch 1 (formats/half/batch come from the config)
DFINE_SEG_CONFIG_DIR=~/dfine_cfg_l uv run dfine export train.root=<root> exp=native_l_<date>

# 5. bench through the harness, next to every rival (idle GPU)
cd <root>/cityscapes-benchmark
uv run --project "$DFINE_SEG_ROOT" python bench_semseg_native.py \
    --families dfine yolo segformer effvit mask2former --fresh 0 \
    --dfine D-FINE-L=<root>/output/models/native_l_<date>/model.engine
uv run --project "$DFINE_SEG_ROOT" python make_results_native.py
```

`exp=` in step 4 points the export at the training run's folder; the default resolves to
today's date. `--fresh 0` reuses stored rival predictions. Run every rival once first; see
[`../README.md`](../README.md).
