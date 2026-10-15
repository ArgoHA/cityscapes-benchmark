"""Run Mask2Former (official FAIR Cityscapes semantic weights) through its OWN detectron2 code path.

  pre   : detectron2 `DefaultPredictor` (BGR->RGB, ResizeShortestEdge(1024, max 2048) = no-op size at
          1024x2048, float32 CHW on CPU), then `MaskFormer.forward` (H2D, mean/std, ImageList pad /32)
  model : TensorRT fp16 engine (backend=trt) or their torch model (backend=torch, compiled
          MultiScaleDeformableAttention CUDA op as their INSTALL.md requires)
  post  : `MaskFormer.forward` eval branch: bilinear mask upsample to the padded input,
          `sem_seg_postprocess`, `semantic_inference` (softmax x sigmoid einsum), then the argmax
          their `SemSegEvaluator` applies

The engine replaces exactly `backbone` + `sem_seg_head` (everything between their preprocessing
and their postprocessing), so `DefaultPredictor.__call__` runs unchanged in both backends.

Plumbing deviations, disclosed in PROTOCOL:
  - Mask2Former ships no ONNX/TensorRT export. The ONNX graph is a TorchScript-exporter trace of
    backbone + sem_seg_head -> (pred_logits, pred_masks), the outputs their post consumes.
  - The compiled deformable-attention op is an autograd Function with no ONNX symbolic, so the trace
    takes their own pure-PyTorch fallback `ms_deform_attn_core_pytorch` (grid_sample), which their
    `MSDeformAttn.forward` already uses whenever the op raises. Same math.
  - Two build-compat edits to their 2022 CUDA source for torch 2.13 (`value.type()` ->
    `value.scalar_type()` in AT_DISPATCH), applied by setup_venvs.sh. No logic change.

Precision: plain fp16 engine, no fp32 pin. Their pixel decoder `forward_features` runs under
`@autocast(enabled=False)`, but pinning it does not help: on R101 (the only variant outside the
0.3 gate, -0.39 vs torch) the whole pixel decoder in fp32 costs ~+40 ms e2e (R50: 66.5 -> 107.6 ms),
the deformable encoder alone in fp32 gives 78.76 (worse than 79.72 unpinned) and the transformer
decoder in fp32 gives 79.73. The loss is diffuse fp16 noise, not that region, so every variant runs as
a standard fp16 build and R101's -0.39 is disclosed. All others are within 0.13.

Repo pinned in third_party/Mask2Former + third_party/detectron2 (setup_venvs.sh). Weights:
artifacts/mask2former/weights (MODEL_ZOO.md, Cityscapes semantic).

  .venv-mask2former/bin/python infer_mask2former.py --variant swin_l --name Mask2Former-Swin-L
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

RIV = Path(__file__).resolve().parent
M2F = RIV / "third_party" / "Mask2Former"
sys.path[:0] = [str(RIV), str(M2F), str(M2F / "mask2former/modeling/pixel_decoder/ops")]

from detectron2.config import get_cfg  # noqa: E402
from detectron2.engine import DefaultPredictor  # noqa: E402
from detectron2.projects.deeplab import add_deeplab_config  # noqa: E402

import dump  # noqa: E402
from build_engine import build  # noqa: E402
from mask2former import add_maskformer2_config  # noqa: E402
from mask2former.modeling.pixel_decoder.ops.modules import ms_deform_attn  # noqa: E402
from model_stats import model_stats  # noqa: E402
from paths import ROOT  # noqa: E402

CFG_DIR = M2F / "configs/cityscapes/semantic-segmentation"
WEIGHTS = RIV / "artifacts" / "mask2former" / "weights"
VARIANTS = {  # variant -> config (MODEL_ZOO.md "Cityscapes Semantic Segmentation")
    "r50": "maskformer2_R50_bs16_90k",
    "r101": "maskformer2_R101_bs16_90k",
    "swin_t": "swin/maskformer2_swin_tiny_bs16_90k",
    "swin_s": "swin/maskformer2_swin_small_bs16_90k",
    "swin_b": "swin/maskformer2_swin_base_IN21k_384_bs16_90k",
    "swin_l": "swin/maskformer2_swin_large_IN21k_384_bs16_90k",
}


def make_cfg(variant: str, device: str = "cuda"):
    cfg = get_cfg()  # = their train_net.setup()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(str(CFG_DIR / f"{VARIANTS[variant]}.yaml"))
    cfg.MODEL.WEIGHTS = str(WEIGHTS / f"{Path(VARIANTS[variant]).name}.pkl")
    cfg.MODEL.DEVICE = device
    cfg.freeze()
    return cfg


class Core(torch.nn.Module):
    """backbone + sem_seg_head on the normalized, padded tensor -> what their post consumes."""

    def __init__(self, model):
        super().__init__()
        self.backbone, self.head = model.backbone, model.sem_seg_head

    def forward(self, x):
        out = self.head(self.backbone(x))
        return out["pred_logits"], out["pred_masks"]


class _NoOp:
    """Makes MSDeformAttn take its own except-branch (pure-PyTorch grid_sample) while tracing."""

    @staticmethod
    def apply(*args):
        raise RuntimeError("tracing: use ms_deform_attn_core_pytorch")


def export_onnx(model, hw, onnx_path: Path):
    if onnx_path.exists():
        return
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dev = next(model.parameters()).device
    real, ms_deform_attn.MSDeformAttnFunction = ms_deform_attn.MSDeformAttnFunction, _NoOp
    try:
        with torch.no_grad():
            torch.onnx.export(
                Core(model).eval(),
                (torch.zeros(1, 3, *hw, device=dev),),
                str(onnx_path),
                opset_version=17,
                input_names=["images"],
                output_names=["pred_logits", "pred_masks"],
                dynamo=False,
            )
    finally:
        ms_deform_attn.MSDeformAttnFunction = real


class TRTCore(torch.nn.Module):
    """Stands in for `backbone` + `sem_seg_head` inside their MaskFormer.forward."""

    def __init__(self, engine_path):
        super().__init__()
        from trt_runner import TRTRunner

        self.runner = TRTRunner(engine_path)
        self.names = list(self.runner.outputs)

    def forward(self, x):
        out = dict(zip(self.names, self.runner(x)))
        return {"pred_logits": out["pred_logits"].float(), "pred_masks": out["pred_masks"].float()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=list(VARIANTS))
    ap.add_argument("--name", required=True)
    ap.add_argument("--imgsz", type=int, nargs=2, default=[1024, 2048], metavar=("H", "W"))
    ap.add_argument("--backend", default="trt", choices=["trt", "torch"])
    ap.add_argument("--export-only", action="store_true", help="write model.onnx and exit")
    ap.add_argument("--device", default="cuda", help="cpu only for --export-only")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    hw = tuple(args.imgsz)
    art = RIV / "artifacts" / "mask2former" / f"{args.variant}_{hw[0]}x{hw[1]}"
    predictor = DefaultPredictor(make_cfg(args.variant, args.device))
    model = predictor.model
    if args.export_only:
        export_onnx(model, hw, art / "model.onnx")
        print(f"wrote {art / 'model.onnx'}")
        sys.exit()

    extra = {
        "variant": args.variant,
        "config": VARIANTS[args.variant],
        "backend": args.backend,
        "input": f"{hw[0]}x{hw[1]}",
        "processor": "detectron2.DefaultPredictor",
    }
    extra.update(model_stats(Core(model), torch.zeros(1, 3, *hw, device="cuda")))

    if args.backend == "trt":
        export_onnx(model, hw, art / "model.onnx")
        engine_path = build(art / "model.onnx", art / "model.engine", half=True)
        extra["engine"] = str(engine_path)
        model.backbone = torch.nn.Identity()
        model.sem_seg_head = TRTCore(engine_path)
        torch.cuda.empty_cache()

    def infer(bgr: np.ndarray):  # timed: their predictor (pre + model + post) + evaluator argmax
        return predictor(bgr)["sem_seg"].argmax(dim=0)

    def extract(seg):  # untimed D2H
        return {"sem_seg": seg.to(torch.uint8).cpu().numpy()}

    dump.run(
        args.name,
        "sem_seg",
        infer,
        extract,
        dump.sem_val_images(args.limit),
        lambda p: cv2.imread(str(p)),  # DefaultPredictor takes BGR, as in their demo
        Path(args.out or ROOT / "output" / "compare_native" / args.name),
        extra=extra,
    )
