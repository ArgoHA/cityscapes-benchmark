"""Run EfficientViT-Seg (official Cityscapes weights) through the EfficientViT repo's OWN code.

Pre/post are imported from their `applications/efficientvit_seg` (the path their demo uses):
  pre   : Compose([Resize((1024, 2048)), ToTensor(ImageNet mean/std)]) on CPU, then .cuda()
  model : TensorRT fp16 engine (backend=trt) or their torch model (backend=torch)
  post  : `efficientvit.models.utils.resize` (bicubic, their default) to the image size -> argmax

Two plumbing deviations, both disclosed in PROTOCOL:
  - Their demo passes `np.ones_like(image)` as a dummy label through ToTensor (a 1024x2048x3 int64
    tensor every frame). We pass a 1x1 dummy instead; the image path is untouched. Favours them.
  - Their `assets/onnx_export.py` builds the model with `pretrained=False` and calls
    `torch.onnx.export` with the exporter that was the default then. We export the *loaded*
    Cityscapes model with the same call (opset 11 + onnxsim) and pin `dynamo=False`, because
    torch>=2.9 switched the default exporter.

Precision: their `LiteMLA.relu_linear_att` runs in fp32 by design (autocast disabled, fp16 input
cast to float): `v @ k^T` sums over every token and overflows fp16 at 1024x2048. The fp32 ONNX graph
does not record that, so a plain `--fp16` build runs it in fp16 and B1/B2/B3/L1 collapse to ~2 mIoU.
The engine keeps exactly those ops (Reshape after the qkv Concat ... Reshape_3) in fp32, as their
code does; everything else (incl. the qkv/aggreg/proj convs) is fp16.

Repo pinned in third_party/efficientvit (see setup_venvs.sh). Weights: HF han-cai/efficientvit-seg.

  .venv-effvit/bin/python -m infer.infer_efficientvit --variant l2 --name EfficientViT-L2
"""

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from common.paths import RIV, ROOT

EVIT = RIV / "third_party" / "efficientvit"
sys.path.insert(0, str(EVIT))
sys.path.insert(0, str(EVIT / "applications" / "efficientvit_seg"))

from efficientvit.models.utils import resize  # noqa: E402
from efficientvit.seg_model_zoo import create_efficientvit_seg_model  # noqa: E402
from eval_efficientvit_seg_model import Resize, ToTensor  # noqa: E402
from torchvision import transforms  # noqa: E402

from common import dump  # noqa: E402
from common.build_engine import build  # noqa: E402
from common.model_stats import model_stats  # noqa: E402


# relu_linear_att ops (see module doc); the qkv/aggreg/proj convs and Concat are not matched
LITEMLA_FP32 = r"context_module/main/(Reshape|Slice|kernel_func|Transpose|Pad|MatMul|Add|Div)"


def export_engine(model, hw, art: Path) -> Path:
    """Their efficientvit.apps.utils.export_onnx, with dynamo=False pinned (see module doc)."""
    from onnxsim import simplify

    onnx_path, engine_path = art / "model.onnx", art / "model.engine"
    if not onnx_path.exists():
        art.mkdir(parents=True, exist_ok=True)
        buf = io.BytesIO()
        with torch.no_grad():
            torch.onnx.export(
                model.eval(),
                torch.rand(1, 3, *hw, device="cuda"),
                buf,
                opset_version=11,
                dynamo=False,
            )
        buf.seek(0)
        m, ok = simplify(onnx.load_model(buf))
        assert ok
        onnx.save(m, str(onnx_path))
    return build(onnx_path, engine_path, half=True, fp32_layers=LITEMLA_FP32)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["b0", "b1", "b2", "b3", "l1", "l2"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--imgsz", type=int, nargs=2, default=[1024, 2048], metavar=("H", "W"))
    ap.add_argument("--backend", default="trt", choices=["trt", "torch"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    hw = tuple(args.imgsz)
    fname = f"efficientvit_seg_{args.variant}_cityscapes.pt"
    weights = hf_hub_download("han-cai/efficientvit-seg", fname)
    model = (
        create_efficientvit_seg_model(
            f"efficientvit-seg-{args.variant}-cityscapes", weight_url=weights
        )
        .cuda()
        .eval()
    )
    extra = {"weights": fname, "backend": args.backend, "input": f"{hw[0]}x{hw[1]}"}
    extra.update(model_stats(model, torch.zeros(1, 3, *hw, device="cuda")))

    transform = transforms.Compose(
        [Resize(hw), ToTensor(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )
    dummy_label = np.zeros((1, 1), np.int64)

    if args.backend == "trt":
        engine_path = export_engine(
            model, hw, RIV / "artifacts" / "efficientvit" / f"{args.variant}_{hw[0]}x{hw[1]}"
        )
        extra["engine"] = str(engine_path)
        del model
        torch.cuda.empty_cache()
        from common.trt_runner import TRTRunner

        runner = TRTRunner(engine_path)

        def forward(x):
            return runner(x)[0]
    else:
        forward = model

    @torch.inference_mode()
    def infer(image: np.ndarray):  # timed: their transform -> model -> their resize + argmax
        data = transform({"data": image, "label": dummy_label})["data"]
        output = forward(torch.unsqueeze(data, dim=0).cuda())
        if output.shape[-2:] != image.shape[:2]:
            output = resize(output, size=image.shape[:2])
        return torch.argmax(output, dim=1)[0]

    def extract(seg):  # untimed D2H
        return {"sem_seg": seg.cpu().numpy().astype(np.uint8)}

    dump.run(
        args.name,
        "sem_seg",
        infer,
        extract,
        dump.sem_val_images(args.limit),
        lambda p: np.array(Image.open(p).convert("RGB")),
        Path(args.out or ROOT / "output" / "compare_native" / args.name),
        extra=extra,
    )
