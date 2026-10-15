"""RF-DETR instance segmentation on Cityscapes through Roboflow's OWN TensorRT path.

Standalone: builds the engine, runs the 500 val images, proves the predictions are correct, and
measures end-to-end latency. Nothing about the model's pre/postprocess is hand-written.

The recipe is the one RF-DETR documents (https://rfdetr.roboflow.com/latest/learn/export/):

    model.export()                                # their ONNX exporter
    rfdetr.export._tensorrt.trtexec(onnx, args)   # their engine builder (trtexec --fp16)
    rfdetr.export.benchmark.TRTInference(engine)  # their engine runner
    rfdetr.models.postprocess.PostProcess         # their postprocessor

and the preprocessing is the exact sequence RFDETR.predict() runs (detr.py:1877-1884):
to_tensor -> H2D -> resize to model resolution on GPU -> normalize with their means/stds.

Correctness oracle: this run's own training-time val metrics (metrics.csv, best EMA epoch 22)
    box mAP@50 0.6124 | mask mAP@50 0.5783 | mask mAP@50-95 0.3293
are recomputed here from the TRT predictions with the same metric the trainer used (torchmetrics
MeanAveragePrecision, faster_coco_eval backend, original 2048x1024 resolution) against the same
val GT (their own YOLO dataset class). If TRT reproduces those numbers, the pipeline is correct.

Latency protocol: identical to D-FINE's src/dl/bench.py -- 10 warmup calls, then per image
cuda.synchronize() -> perf_counter() -> the full end-to-end call -> cuda.synchronize(); reported
as mean(latency[1:]). Image decode and the device->host copy of the results sit outside the clock,
exactly as in bench.py.

Runs in `.venv-rfdetr-trtexec`, NOT the benchmark's `.venv-rfdetr`. TRT engines are version-locked
and their `trtexec` helper shells out to the `trtexec` binary on PATH (system TRT 10.11), so the
runtime that loads the engine must be 10.11 too. `.venv-rfdetr` is pinned to 10.13.3.9 (what every
engine in the benchmark proper is built at) and cannot deserialize it. With the versions matched,
their helper needs no modification.

    .venv-rfdetr-trtexec/bin/python rfdetr_trt.py                # full run: 500 val images
    .venv-rfdetr-trtexec/bin/python rfdetr_trt.py --limit 25     # smoke test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import tensorrt
import torch
import torch.nn.functional as NF  # noqa: N812
import torchvision.transforms.functional as TF  # noqa: N812
from PIL import Image
from tqdm import tqdm

# UPSTREAM BUG 1 -- rfdetr/export/benchmark.py:28 does `import _tensorrt as trt`. There is no
# top-level `_tensorrt` module (their rfdetr/export/_tensorrt.py is a trtexec CLI wrapper with no
# Logger/Runtime/nptype), so the import always fails, `trt` is always None, and TRTInference dies
# with "TensorRT is not installed" even on a machine where it is. Aliasing the real module before
# importing benchmark makes their class work as intended. Must run before the rfdetr import.
sys.modules.setdefault("_tensorrt", tensorrt)

from rfdetr import RFDETRSegMedium  # noqa: E402
from rfdetr.datasets import build_dataset  # noqa: E402
from rfdetr.export._tensorrt import trtexec  # noqa: E402
from rfdetr.export.benchmark import TRTInference  # noqa: E402
from rfdetr.utilities.box_ops import box_cxcywh_to_xyxy  # noqa: E402
from torchmetrics.detection import MeanAveragePrecision  # noqa: E402

from paths import ROOT  # noqa: E402

RUN_DIR = ROOT / "output" / "models" / "rfdetr_seg_medium"
DEVICE = "cuda:0"


def build_engine(model, out_dir: Path) -> Path:
    """RF-DETR's documented export path: their ONNX exporter, then their trtexec helper."""
    onnx_path = Path(model.export(output_dir=str(out_dir), verbose=False))
    engine_path = onnx_path.with_suffix(".engine")
    if not engine_path.exists():
        # their helper shells out to `trtexec` on PATH with --fp16 and a 4 GB workspace, and also
        # runs its own engine-only benchmark (10 s). The venv's TensorRT python runtime is pinned
        # to the same 10.11 build as the system trtexec, otherwise the engine it writes cannot be
        # deserialized by the runtime that has to load it (TRT engines are version-locked).
        trtexec(str(onnx_path), Namespace(verbose=False, profile=False, dry_run=False))
    return engine_path


def make_infer(trt_model, post, means, stds, resolution: int, conf: float):
    """RFDETR.predict() with the torch forward swapped for the TRT engine. Everything else theirs."""

    def infer(img: Image.Image):
        # --- preprocess: detr.py:1877-1884, verbatim (resize runs on the GPU, as in predict())
        t = TF.to_tensor(img).to(DEVICE)
        batch = torch.stack([TF.resize(t, [resolution, resolution])])
        batch = TF.normalize(batch, means, stds)

        # UPSTREAM BUG 2 -- TRTInference.run_sync hands blob[name].data_ptr() straight to TRT,
        # which reads it as contiguous NCHW. A non-contiguous input tensor is silently read as
        # scrambled memory (garbage logits, no error). predict()'s own preprocess happens to
        # produce a contiguous tensor; assert it rather than paper over it with .contiguous().
        assert batch.is_contiguous(), "TRTInference would silently read scrambled memory"

        outputs = trt_model({"input": batch})

        # --- postprocess: their PostProcess (top-k -> xyxy at orig size, masks -> orig size)
        preds = {
            "pred_logits": outputs["labels"],
            "pred_boxes": outputs["dets"],
            "pred_masks": outputs["masks"],
        }
        h, w = img.size[1], img.size[0]
        target_sizes = torch.tensor([[h, w]], device=DEVICE)
        result = post(preds, target_sizes)[0]

        # --- threshold, as predict() does (detr.py:1983). Kept on the clock: it indexes the
        # [num_select, 1, H, W] mask tensor, which is real work, and every rival filters on-clock.
        keep = result["scores"] > conf
        filtered = {k: v[keep] for k, v in result.items()}
        return filtered, result  # unfiltered result feeds the mAP oracle, off the clock

    return infer


def _box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix between two sets of xyxy boxes."""
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), axis=2)
    area = lambda x: np.prod(np.clip(x[:, 2:] - x[:, :2], 0, None), axis=1)  # noqa: E731
    return inter / np.maximum(area(a)[:, None] + area(b)[None, :] - inter, 1e-9)


def check_parity(model, infer, image_paths, conf: float, n: int = 20) -> dict:
    """TRT vs their own PyTorch predict() on the same images: scores, boxes, masks.

    fp16 TRT and fp32 torch reorder near-tied queries, so detections are matched by box IoU
    (same class), not by rank.
    """
    d_score, d_box, mask_iou = [], [], []
    n_trt = n_torch = n_matched = 0
    for p in tqdm(image_paths[:n], desc="parity vs torch predict()"):
        img = Image.open(p).convert("RGB")
        det = model.predict(
            img, threshold=conf, include_source_image=False
        )  # their fp32 torch path
        trt_res, _ = infer(img)

        t_scores = trt_res["scores"].float().cpu().numpy()
        t_boxes = trt_res["boxes"].float().cpu().numpy()
        t_labels = trt_res["labels"].cpu().numpy()
        t_masks = trt_res["masks"].squeeze(1).cpu().numpy()
        n_trt += len(t_scores)
        n_torch += len(det)
        if not len(det) or not len(t_scores):
            continue

        iou = _box_iou(det.xyxy.astype(np.float32), t_boxes)
        iou[det.class_id[:, None] != t_labels[None, :]] = 0.0  # only match same class
        for i in np.argsort(-det.confidence):  # greedy, highest-confidence torch det first
            j = int(np.argmax(iou[i]))
            if iou[i, j] < 0.5:
                continue
            n_matched += 1
            iou[:, j] = 0.0  # consume
            d_score.append(abs(float(det.confidence[i]) - float(t_scores[j])))
            d_box.append(float(np.abs(det.xyxy[i] - t_boxes[j]).max()))
            a, b = det.mask[i].astype(bool), t_masks[j].astype(bool)
            union = (a | b).sum()
            mask_iou.append(float((a & b).sum() / union) if union else 1.0)

    return {
        "images": min(n, len(image_paths)),
        "n_det_torch": n_torch,
        "n_det_trt": n_trt,
        "matched": n_matched,
        "max_abs_score_delta": round(float(np.max(d_score)), 5) if d_score else None,
        "max_abs_box_delta_px": round(float(np.max(d_box)), 3) if d_box else None,
        "mean_mask_iou": round(float(np.mean(mask_iou)), 5) if mask_iou else None,
        "min_mask_iou": round(float(np.min(mask_iou)), 5) if mask_iou else None,
    }


def profile_stages(
    trt_model, post, means, stds, resolution: int, conf: float, paths, n: int = 50
) -> dict:
    """Split the end-to-end call into preprocess / engine / postprocess, same sync discipline."""

    def clock(fn):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        return out, (time.perf_counter() - t0) * 1000

    pre, eng, pp, thr = [], [], [], []
    for p in tqdm(paths[:n], desc="stage profile"):
        img = Image.open(p).convert("RGB")
        h, w = img.size[1], img.size[0]

        def _pre(img=img):
            t = TF.to_tensor(img).to(DEVICE)
            return TF.normalize(torch.stack([TF.resize(t, [resolution, resolution])]), means, stds)

        batch, t_pre = clock(_pre)
        outputs, t_eng = clock(lambda: trt_model({"input": batch}))
        packed = {
            "pred_logits": outputs["labels"],
            "pred_boxes": outputs["dets"],
            "pred_masks": outputs["masks"],
        }
        sizes = torch.tensor([[h, w]], device=DEVICE)
        result, t_pp = clock(lambda: post(packed, sizes)[0])
        _, t_thr = clock(lambda: {k: v[result["scores"] > conf] for k, v in result.items()})
        pre.append(t_pre), eng.append(t_eng), pp.append(t_pp), thr.append(t_thr)

    mean = lambda x: round(float(np.mean(x[1:])), 2)  # noqa: E731
    return {
        "preprocess_ms": mean(pre),
        "engine_ms": mean(eng),
        "postprocess_ms": mean(pp),
        "threshold_ms": mean(thr),
        "sum_ms": round(mean(pre) + mean(eng) + mean(pp) + mean(thr), 2),
    }


def gt_from_target(target: dict) -> dict:
    """Val GT the way the trainer's COCOEvalCallback builds it (coco_eval.py:_convert_targets)."""
    h, w = target["orig_size"].tolist()
    scale = target["boxes"].new_tensor([w, h, w, h])
    entry = {"boxes": box_cxcywh_to_xyxy(target["boxes"]) * scale, "labels": target["labels"]}
    masks = target["masks"].bool()
    if masks.shape[-2:] != (int(h), int(w)):  # GT masks are stored at the model resolution
        masks = (
            NF.interpolate(masks.float().unsqueeze(1), size=(int(h), int(w)), mode="nearest")
            .squeeze(1)
            .bool()
        )
    entry["masks"] = masks
    return {
        k: v.to(DEVICE) for k, v in entry.items()
    }  # preds live on GPU; the metric needs one device


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(RUN_DIR / "checkpoint_best_ema.pth"))
    ap.add_argument("--out", default=str(ROOT / "output" / "compare" / "rfdetr_seg_trt"))
    ap.add_argument(
        "--conf", type=float, default=0.5, help="operating point for latency, as in bench.py"
    )
    ap.add_argument("--limit", type=int, default=0, help="first N val images only (smoke test)")
    ap.add_argument("--parity", type=int, default=20, help="images for the TRT-vs-torch check")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((RUN_DIR / "training_config.json").read_text())
    ds_args = Namespace(**{**cfg["train_config"], **cfg["model_config"]})

    model = RFDETRSegMedium(pretrain_weights=args.ckpt)
    resolution = model.model.resolution
    print(
        f"resolution={resolution} num_select={model.model_config.num_select} classes={model.class_names}"
    )

    engine_path = build_engine(model, out_dir)
    trt_model = TRTInference(engine_path=str(engine_path), device=DEVICE, sync_mode=True)
    post = model.model.postprocess  # their PostProcess instance, with this model's num_select
    model_means, model_stds = model.means, model.stds
    infer = make_infer(trt_model, post, model_means, model_stds, resolution, args.conf)

    # val GT + image paths from their own dataset class (same one the trainer validated on)
    ds = build_dataset("val", ds_args, resolution)
    image_paths = [Path(ds.coco.loadImgs(i)[0]["file_name"]) for i in ds.ids]
    idxs = list(range(len(ds)))[: args.limit] if args.limit else list(range(len(ds)))
    print(f"val images: {len(idxs)}")

    parity = check_parity(model, infer, image_paths, args.conf, args.parity)
    print("\nparity vs their PyTorch predict():", json.dumps(parity, indent=2))

    del model  # deployment conditions for the timed pass: engine only, no torch model resident
    torch.cuda.empty_cache()

    metric = MeanAveragePrecision(
        iou_type=["bbox", "segm"],  # the trainer's exact metric config (coco_eval.py:164-176)
        class_metrics=False,
        max_detection_thresholds=[1, 10, cfg["train_config"]["eval_max_dets"]],
        backend="faster_coco_eval",
    )

    warm = Image.open(image_paths[idxs[0]]).convert("RGB")
    for _ in range(10):
        infer(warm)
    torch.cuda.synchronize()

    latency = []
    for i in tqdm(idxs, desc="TRT e2e"):
        img = Image.open(image_paths[i]).convert("RGB")  # decode is off the clock, as in bench.py

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _, full = infer(img)
        torch.cuda.synchronize()
        latency.append((time.perf_counter() - t0) * 1000)

        _, target = ds[i]  # GT (off the clock)
        pred = {
            "boxes": full["boxes"],
            "scores": full["scores"],
            "labels": full["labels"],
            "masks": full["masks"].squeeze(1),
        }  # PostProcess emits [K,1,H,W]; the metric wants [K,H,W]
        metric.update([pred], [gt_from_target(target)])

    stages = profile_stages(
        trt_model,
        post,
        model_means,
        model_stds,
        resolution,
        args.conf,
        [image_paths[i] for i in idxs],
        n=args.parity,
    )

    m = {k: float(v) for k, v in metric.compute().items() if v.numel() == 1}
    result = {
        "engine": str(engine_path),
        "resolution": resolution,
        "n_images": len(idxs),
        "conf": args.conf,
        "latency_ms": round(float(np.mean(latency[1:])), 2),
        "latency_p50_ms": round(float(np.median(latency[1:])), 2),
        "stage_breakdown": stages,
        "box_mAP_50": round(m["bbox_map_50"], 4),
        "box_mAP_50_95": round(m["bbox_map"], 4),
        "mask_mAP_50": round(m["segm_map_50"], 4),
        "mask_mAP_50_95": round(m["segm_map"], 4),
        "parity_vs_torch": parity,
    }
    (out_dir / "trt_result.json").write_text(json.dumps(result, indent=2))
    print("\n" + json.dumps(result, indent=2))
    print(
        "\ntraining-time oracle (metrics.csv, best EMA epoch): "
        "box mAP@50 0.6124 | mask mAP@50 0.5783 | mask mAP@50-95 0.3293"
    )


if __name__ == "__main__":
    main()
