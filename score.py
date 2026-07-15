"""Score every framework's dumped predictions with D-FINE's OWN validator.

One scorer, one GT pipeline, one metric definition for all three frameworks. GT is built with
D-FINE's BenchLoader exactly as src/dl/bench.py does (boxes rescaled to original resolution,
instance masks rasterized from original-resolution polygons, sem_seg compared at original
resolution), so the numbers are directly comparable to the existing bench_metrics.csv.

  uv run --project /home/argo/Desktop/Projects/D-FINE-seg python score.py \
      --task detect --cfg <model_dir>/config.yaml \
      --preds <compare>/dfine_s_det <compare>/yolo26s_det <compare>/rfdetr_s_det

Run in the D-FINE venv.
"""

import argparse
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from tabulate import tabulate
from tqdm import tqdm

DFINE = Path("/home/argo/Desktop/Projects/D-FINE-seg")
sys.path.insert(0, str(DFINE))

from src.dl.bench import BenchLoader  # noqa: E402
from src.dl.utils import encode_sample_masks_to_rle, poly_abs_to_mask, process_boxes  # noqa: E402
from src.dl.validator import SemSegValidator, Validator  # noqa: E402

IOU_THRESH = 0.5


def build_gt(cfg):
    """GT exactly as bench.py builds it, keyed by image filename."""
    data_path = Path(cfg.train.data_path)
    loader, _ = BenchLoader(
        root_path=data_path,
        img_size=tuple(cfg.train.img_size),
        batch_size=1,
        num_workers=1,
        cfg=cfg,
        debug_img_processing=False,
    ).build_dataloaders()

    gt = {}
    for _, targets, img_paths in tqdm(loader, total=len(loader), desc="GT"):
        for img_path, t in zip(img_paths, targets):
            boxes = process_boxes(
                t["boxes"][None],
                tuple(cfg.train.img_size),
                t["orig_size"][None],
                cfg.train.keep_ratio,
                cfg.train.device,
            )[0].cpu()
            d = {"boxes": boxes, "labels": t["labels"].int()}
            if "masks" in t:
                H0, W0 = int(t["orig_size"][0]), int(t["orig_size"][1])
                polys = t.get("polys")
                d["masks"] = (
                    torch.from_numpy(
                        np.stack(
                            [
                                poly_abs_to_mask(p, H0, W0)
                                if getattr(p, "size", 0)
                                else np.zeros((H0, W0), np.uint8)
                                for p in polys
                            ]
                        )
                    ).to(torch.uint8)
                    if polys
                    else torch.zeros((0, H0, W0), dtype=torch.uint8)
                )
            encode_sample_masks_to_rle(d)
            gt[str(img_path)] = d  # loader yields PosixPath; dumps key on the filename string
    return gt


def _align(blob, gt):
    all_gt, all_preds = [], []
    for rec in blob["preds"]:
        g = gt.get(rec["file"])
        if g is None:
            raise KeyError(f"{rec['file']} not in GT")
        all_gt.append(g)
        p = {
            "boxes": torch.from_numpy(rec["boxes"]),
            "labels": torch.from_numpy(rec["labels"]).int(),
            "scores": torch.from_numpy(rec["scores"]),
        }
        if "masks_rle" in rec:
            p["masks_rle"] = rec["masks_rle"]
            p["masks_size"] = rec["masks_size"]
        all_preds.append(p)
    return all_gt, all_preds


def _filter(all_preds, conf):
    """Validator does NOT filter by conf_thresh - it scores whatever it's handed (bench.py gets
    away with this because TRT_model already filters inside the wrapper). Our dumps are made at a
    low conf so the sweep has room, so we must filter here, keeping masks_rle in sync."""
    out = []
    for p in all_preds:
        keep = p["scores"] >= conf
        idx = torch.nonzero(keep, as_tuple=True)[0].tolist()
        q = {"boxes": p["boxes"][keep], "labels": p["labels"][keep], "scores": p["scores"][keep]}
        if "masks_rle" in p:
            q["masks_rle"] = [p["masks_rle"][i] for i in idx]
            q["masks_size"] = p["masks_size"]
        out.append(q)
    return out


def _metrics_at(all_gt, all_preds, label_to_name, conf, boxes_only=False):
    """boxes_only: drop masks entirely. Used for the threshold SWEEP, because a mask-based sweep
    means RLE-decoding + mask-IoU-matching ~300 preds/image at 1024x2048, 500 images, 19 times
    over - hours. D-FINE's own optimal_thresh sweep is boxes-only for the same reason
    (validator.py: `_compute_main_metrics(filtered, ignore_masks=True)`), and the threshold being
    swept is the *detection* score, so the box- and mask-optimal thresholds coincide closely.
    The FINAL reported metrics are still full mask-based, computed at the chosen t*."""
    preds = _filter(all_preds, conf)
    if boxes_only:
        preds = [{k: v for k, v in p.items() if k not in ("masks_rle", "masks_size")}
                 for p in preds]
    v = Validator(
        all_gt, preds, conf_thresh=conf, iou_thresh=IOU_THRESH,
        label_to_name=label_to_name, compute_maps=False,
    )
    return v.compute_metrics(extended=False, ignore_masks=boxes_only)


def score_boxes(blob, gt, label_to_name, conf, sweep):
    """Score at a fixed conf, and (if the dump has low-conf preds) at this model's OWN val-optimal
    threshold. Each framework's scores are calibrated differently - YOLO26 is NMS-free, RF-DETR is
    DETR-sigmoid, D-FINE has its own head - so a single fixed threshold flatters some and punishes
    others for reasons unrelated to quality. The per-model optimum is the like-for-like number.

    Requires the dump to have been made at a LOW conf: the sweep can never go below the threshold
    the predictions were already filtered at.
    """
    all_gt, all_preds = _align(blob, gt)
    # sweep stage only needs opt_thresh -> keep everything boxes-only and cheap.
    # the final (non-sweep) call is the one that computes full mask-based metrics at t*.
    m = _metrics_at(all_gt, all_preds, label_to_name, conf, boxes_only=sweep)

    if not sweep:
        return m

    dump_conf = blob.get("conf", 0.0)
    best = (-1.0, None)
    for t in np.round(np.arange(0.05, 0.96, 0.05), 2):
        if t < dump_conf:  # unreachable - preds were already filtered above this
            continue
        mt = _metrics_at(all_gt, all_preds, label_to_name, float(t), boxes_only=True)
        if mt["f1"] > best[0]:
            best = (mt["f1"], float(t))

    m = {f"{k}@{conf}": v for k, v in m.items()}
    m["opt_thresh"] = best[1]
    return m


def score_sem(blob, cfg, preds_dir):
    data_path = Path(cfg.train.data_path)
    v = SemSegValidator(
        len(cfg.train.label_to_name),
        cfg.train.label_to_name,
        int(cfg.train.sem_seg.ignore_index),
    )
    for rec in tqdm(blob["preds"], desc="sem_seg"):
        stem = Path(rec["file"]).stem
        pred = cv2.imread(str(preds_dir / "sem_seg" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(str(data_path / "labels" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        v.update(torch.from_numpy(pred), torch.from_numpy(gt))
    return v.compute_metrics()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["detect", "segment", "sem_seg"])
    ap.add_argument("--cfg", required=True, help="a D-FINE config.yaml for this task (GT source)")
    ap.add_argument("--preds", nargs="+", required=True, help="dirs containing preds.pkl")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--sweep", type=int, default=1, help="also report F1 at each model's own optimum")
    ap.add_argument("--latency-from", nargs="*", default=[],
                    help="dirs holding the conf=0.5 latency pass (accuracy dumps use a low conf)")
    ap.add_argument("--out", default=None)
    # the sem_seg run's frozen config points at data/, but the dataset was later reorganised into
    # data/dataset/ (when dataset_det / dataset_seg were split out). Override rather than guess.
    ap.add_argument("--data-path", default=None)
    args = ap.parse_args()

    cfg = OmegaConf.load(args.cfg)
    if args.data_path:
        cfg.train.data_path = args.data_path
    dp = Path(str(cfg.train.data_path))
    if not (dp / "images").exists():
        sys.exit(f"data_path has no images/: {dp} — pass --data-path")

    gt = None if args.task == "sem_seg" else build_gt(cfg)

    # latency must come from a pass run at the production operating point, not the low-conf
    # accuracy pass (more surviving boxes = more postprocess work = inflated latency)
    lat = {}
    for d in args.latency_from:
        b = pickle.load(open(Path(d) / "preds.pkl", "rb"))
        lat[b["model"]] = b["latency_ms"]

    rows = {}
    for d in args.preds:
        d = Path(d)
        blob = pickle.load(open(d / "preds.pkl", "rb"))
        m = (
            score_sem(blob, cfg, d)
            if args.task == "sem_seg"
            else score_boxes(blob, gt, cfg.train.label_to_name, args.conf, bool(args.sweep))
        )
        m["latency"] = round(lat.get(blob["model"], blob["latency_ms"]), 2)
        rows[blob["model"]] = m

    df = pd.DataFrame.from_dict(rows, orient="index").round(3)
    out = args.out or f"/home/argo/Desktop/Projects/cityscapes/output/compare/{args.task}.csv"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out)
    print(f"\n=== {args.task} (conf={args.conf}, iou={IOU_THRESH}) ===")
    print(tabulate(df, headers="keys", tablefmt="github", showindex=True))
    print(f"\nwrote {out}")
