"""Run SegFormer (official NVIDIA Cityscapes weights) through HF transformers' OWN pre/postprocess.

  pre   : `AutoImageProcessor` (SegformerImageProcessor) with size=1024x2048 - see the trap below
  model : TensorRT fp16 engine (backend=trt) or the HF torch model (backend=torch)
  post  : `processor.post_process_semantic_segmentation(target_sizes=...)`
          (bilinear logits upsample to the original size, then argmax)

Trap: the processor's default `size` is 512x512 (1024x1024 for b1). Left alone it squashes a
2048x1024 frame into a square. We pass size={"height": 1024, "width": 2048}, a processor config
argument, not hand-written preprocessing.

transformers 5.x image processors are torchvision-backed and take `device=`; `--pre-device cuda`
(the default) runs that same preprocessing on the GPU, `cpu` is the processor's default device.

HF ships no TensorRT runtime and no ONNX exporter inside transformers, so the ONNX graph is a plain
TorchScript-exporter trace of `model(pixel_values).logits` (what optimum's exporter does for this
architecture) and the engine runs through trt_runner.TRTRunner. Build flags as every other rival.

--mode slide (torch only) is the reproduction gate for the paper numbers, which use mmseg's sliding
window (1024x1024 crops, stride 768). It is never used for the benchmark table.

  .venv-segformer/bin/python -m infer.infer_segformer --variant b2 --name SegFormer-B2
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from PIL import Image
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
from transformers.modeling_outputs import SemanticSegmenterOutput

from common import dump
from common.build_engine import build
from common.model_stats import model_stats

from common.paths import RIV, ROOT


class Logits(torch.nn.Module):
    """Export wrapper: the HF model returns a ModelOutput; the engine needs a plain tensor."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m(pixel_values=x).logits


def export_engine(model, hw, art: Path) -> Path:
    onnx_path, engine_path = art / "model.onnx", art / "model.engine"
    if not onnx_path.exists():
        art.mkdir(parents=True, exist_ok=True)
        x = torch.zeros(1, 3, *hw, device="cuda")
        with torch.no_grad():  # B5 at 1024x2048 OOMs 16 GB if the trace keeps autograd state
            torch.onnx.export(
                Logits(model).eval(),
                (x,),
                str(onnx_path),
                opset_version=17,
                input_names=["pixel_values"],
                output_names=["logits"],
                dynamo=False,
            )
    return build(onnx_path, engine_path, half=True)


@torch.no_grad()
def slide_logits(model, x, crop=1024, stride=768):
    """mmseg `EncoderDecoder.slide_inference` (whole-image test cfg of the SegFormer release)."""
    _, _, H, W = x.shape
    nc = model.config.num_labels
    acc = torch.zeros(1, nc, H, W, device=x.device)
    cnt = torch.zeros(1, 1, H, W, device=x.device)
    h_grids = max(H - crop + stride - 1, 0) // stride + 1
    w_grids = max(W - crop + stride - 1, 0) // stride + 1
    for hi in range(h_grids):
        for wi in range(w_grids):
            y1, x1 = hi * stride, wi * stride
            y2, x2 = min(y1 + crop, H), min(x1 + crop, W)
            y1, x1 = max(y2 - crop, 0), max(x2 - crop, 0)
            lg = model(pixel_values=x[:, :, y1:y2, x1:x2]).logits
            lg = F.interpolate(lg, size=(y2 - y1, x2 - x1), mode="bilinear", align_corners=False)
            acc[:, :, y1:y2, x1:x2] += lg
            cnt[:, :, y1:y2, x1:x2] += 1
    return acc / cnt


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["b0", "b1", "b2", "b3", "b4", "b5"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--imgsz", type=int, nargs=2, default=[1024, 2048], metavar=("H", "W"))
    ap.add_argument("--backend", default="trt", choices=["trt", "torch"])
    ap.add_argument("--mode", default="whole", choices=["whole", "slide"])
    ap.add_argument("--pre-device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if args.mode == "slide" and args.backend != "torch":
        ap.error("--mode slide is the torch reproduction gate only")

    hw = tuple(args.imgsz)
    repo = f"nvidia/segformer-{args.variant}-finetuned-cityscapes-1024-1024"
    processor = AutoImageProcessor.from_pretrained(repo)
    model = SegformerForSemanticSegmentation.from_pretrained(repo).eval().cuda()
    extra = {
        "repo": repo,
        "backend": args.backend,
        "mode": args.mode,
        "input": f"{hw[0]}x{hw[1]}",
        "pre_device": args.pre_device,
        "processor": type(processor).__name__,
    }
    extra.update(model_stats(Logits(model), torch.zeros(1, 3, *hw, device="cuda")))

    pre_kw = dict(
        size={"height": hw[0], "width": hw[1]}, return_tensors="pt", device=args.pre_device
    )
    if args.backend == "trt":
        engine_path = export_engine(
            model, hw, RIV / "artifacts" / "segformer" / f"{args.variant}_{hw[0]}x{hw[1]}"
        )
        extra["engine"] = str(engine_path)
        del model
        torch.cuda.empty_cache()
        from common.trt_runner import TRTRunner

        runner = TRTRunner(engine_path)

        def forward(px):
            return runner(px)[0]
    elif args.mode == "slide":

        def forward(px):
            return slide_logits(model, px)
    else:

        def forward(px):
            return model(pixel_values=px).logits

    @torch.inference_mode()
    def infer(img: Image.Image):  # timed: HF processor -> model -> HF postprocess
        px = processor(images=img, **pre_kw)["pixel_values"].to("cuda", non_blocking=True)
        out = SemanticSegmenterOutput(logits=forward(px))
        return processor.post_process_semantic_segmentation(
            out, target_sizes=[(img.height, img.width)]
        )[0]

    def extract(seg):  # untimed D2H
        return {"sem_seg": seg.cpu().numpy().astype(np.uint8)}

    dump.run(
        args.name,
        "sem_seg",
        infer,
        extract,
        dump.sem_val_images(args.limit),
        lambda p: Image.open(p).convert("RGB"),
        Path(args.out or ROOT / "output" / "compare_native" / args.name),
        extra=extra,
    )
