#!/usr/bin/env bash
# Venvs for the native-res sem_seg bench (PROTOCOL.md §11). One per framework, all on
# torch 2.13 + TensorRT 10.13.3.9 so every engine is built and run by the same TRT version.
# The D-FINE side (scorer, D-FINE inference, engine timing) uses the D-FINE-seg uv env.
set -euo pipefail
cd "$(dirname "$0")"

COMMON=(torch==2.13.0 torchvision==0.28.0 tensorrt-cu12==10.13.3.9 opencv-python-headless
        pandas tabulate tqdm pillow numpy)

# YOLO26-sem: ultralytics 8.4.92 (see repro_yolo_val.py for why the version matters). Weights
# yolo26{n..x}-sem.pt are ultralytics release assets, downloaded on first use. An existing
# .venv-yolo (the round-1 one) is kept as is.
if [ ! -d .venv-yolo ]; then
  uv venv -p 3.12 .venv-yolo
  uv pip install -p .venv-yolo/bin/python "${COMMON[@]}" onnx "ultralytics==8.4.92"
fi

# SegFormer: HF transformers (pre/post + weights). ONNX via the TorchScript exporter.
uv venv -p 3.12 .venv-segformer
uv pip install -p .venv-segformer/bin/python "${COMMON[@]}" onnx "transformers==5.19.0"

# EfficientViT: their repo at a pinned commit, plus what its seg model + ONNX export import.
# onnxsim 0.4.36 needs onnx<=1.17 (newer onnx rejects its transient Constant attribute).
EVIT_COMMIT=de7d7733cc0329f391b33f1f459271562ec27bd5
if [ ! -d third_party/efficientvit ]; then
  git clone https://github.com/mit-han-lab/efficientvit.git third_party/efficientvit
fi
git -C third_party/efficientvit checkout -q "$EVIT_COMMIT"
uv venv -p 3.12 .venv-effvit
uv pip install -p .venv-effvit/bin/python "${COMMON[@]}" "onnx==1.17.0" "onnxsim==0.4.36" \
  onnxruntime timm einops omegaconf huggingface-hub scipy matplotlib torchprofile \
  "git+https://github.com/facebookresearch/segment-anything.git"

# Mask2Former: their repo + detectron2 at pinned commits, weights from their MODEL_ZOO.md.
# torch 2.13 has no cu128 wheel; cu129 matches the system nvcc 12.8 closely enough to compile their
# MultiScaleDeformableAttention op (cu130 does not). TRT is the same 10.13.3.9 as everywhere else.
D2_COMMIT=1e3e13bbf607b54f62205c4c33922521822fb298
M2F_COMMIT=9b0651c6c1d5b3af2e6da0589b719c514ec0d69a
[ -d third_party/detectron2 ] || git clone https://github.com/facebookresearch/detectron2.git third_party/detectron2
[ -d third_party/Mask2Former ] || git clone https://github.com/facebookresearch/Mask2Former.git third_party/Mask2Former
git -C third_party/detectron2 checkout -q "$D2_COMMIT"
git -C third_party/Mask2Former checkout -q "$M2F_COMMIT"
uv venv -p 3.12 .venv-mask2former
uv pip install -p .venv-mask2former/bin/python --index-url https://download.pytorch.org/whl/cu129 \
  --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match \
  "torch==2.13.0+cu129" "torchvision==0.28.0+cu129"
uv pip install -p .venv-mask2former/bin/python tensorrt-cu12==10.13.3.9 opencv-python-headless \
  pandas tabulate tqdm pillow "numpy<2.4" onnx onnxruntime cython scipy shapely timm h5py submitit \
  scikit-image setuptools wheel ninja fvcore iopath pycocotools
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH TORCH_CUDA_ARCH_LIST=12.0
uv pip install -p .venv-mask2former/bin/python --no-build-isolation -e third_party/detectron2
# build-compat for torch 2.13 (removed DeprecatedTypeProperties overload); no logic change
sed -i 's/AT_DISPATCH_FLOATING_TYPES(value\.type()/AT_DISPATCH_FLOATING_TYPES(value.scalar_type()/' \
  third_party/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_attn_cuda.cu
(cd third_party/Mask2Former/mask2former/modeling/pixel_decoder/ops &&
  FORCE_CUDA=1 ../../../../../../.venv-mask2former/bin/python setup.py build_ext --inplace)
W=artifacts/mask2former/weights
B=https://dl.fbaipublicfiles.com/maskformer/mask2former/cityscapes/semantic
mkdir -p "$W"
for p in maskformer2_R50_bs16_90k/model_final_cc1b1f.pkl maskformer2_R101_bs16_90k/model_final_257ce8.pkl \
         maskformer2_swin_tiny_bs16_90k/model_final_2d58d4.pkl maskformer2_swin_small_bs16_90k/model_final_fa26ae.pkl \
         maskformer2_swin_base_IN21k_384_bs16_90k/model_final_1c6b65.pkl \
         maskformer2_swin_large_IN21k_384_bs16_90k/model_final_17c1ee.pkl; do
  [ -s "$W/$(dirname "$p").pkl" ] || curl -fL -o "$W/$(dirname "$p").pkl" "$B/$p"
done
