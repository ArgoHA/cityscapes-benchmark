"""Build YOLO26 / RF-DETR dataset layouts from the existing D-FINE cityscapes datasets.

Each framework gets the data in ITS OWN native layout, at ORIGINAL resolution (2048x1024), and
does its own preprocessing (YOLO letterboxes, RF-DETR square-resizes, D-FINE square-resizes).
We are comparing architecture + framework end-to-end, so no geometry is normalized here.
Images are symlinked - nothing is re-encoded or resized on disk.

Sources (read-only):
  data/dataset       images/ + labels/*.txt (bbox)      + train.csv/val.csv   (detection; owns the images)
  data/dataset_seg   images/ + labels/*.txt (polygons)  + train.csv/val.csv
  data/dataset_sem   images/ + labels/*.png (trainIds)  + train.csv/val.csv

data/dataset_sem (D-FINE-seg's sem_seg layout; the native sem_seg benchmark needs only this and
yolo_sem) is built from the official Cityscapes zips:
  python prep_datasets.py --only dataset_sem yolo_sem --raw /path/to/cityscapes_raw
where cityscapes_raw holds the unzipped leftImg8bit_trainvaltest.zip and gtFine_trainvaltest.zip.

Outputs (under data/):
  yolo_det/{images,labels}/{train,val}      + yolo_det.yaml     (ultralytics detect)
  yolo_seg/{images,labels}/{train,val}      + yolo_seg.yaml     (ultralytics segment)
  yolo_sem/{images,masks}/{train,val}       + yolo_sem.yaml     (ultralytics semantic)
  rfdetr_det/{train,valid}/{images,labels}  + data.yaml         (rfdetr, YOLO-format autodetect)
  rfdetr_seg/{train,valid}/{images,labels}  + data.yaml         (polygons -> masks internally)
"""

import argparse
import shutil
from pathlib import Path

import yaml

from common.paths import DATA

THING_CLASSES = {
    0: "person",
    1: "rider",
    2: "car",
    3: "truck",
    4: "bus",
    5: "train",
    6: "motorcycle",
    7: "bicycle",
}
SEM_CLASSES = {
    0: "road",
    1: "sidewalk",
    2: "building",
    3: "wall",
    4: "fence",
    5: "pole",
    6: "traffic-light",
    7: "traffic-sign",
    8: "vegetation",
    9: "terrain",
    10: "sky",
    11: "person",
    12: "rider",
    13: "car",
    14: "truck",
    15: "bus",
    16: "train",
    17: "motorcycle",
    18: "bicycle",
}

# Cityscapes labelId -> 19 trainIds (the cityscapesScripts mapping); every other id -> 255 (ignore)
LABEL_TO_TRAIN = {
    7: 0,
    8: 1,
    11: 2,
    12: 3,
    13: 4,
    17: 5,
    19: 6,
    20: 7,
    21: 8,
    22: 9,
    23: 10,
    24: 11,
    25: 12,
    26: 13,
    27: 14,
    28: 15,
    31: 16,
    32: 17,
    33: 18,
}


def read_split(ds: Path, split: str) -> list[str]:
    return [ln.strip() for ln in (ds / f"{split}.csv").read_text().splitlines() if ln.strip()]


def link(src: Path, dst: Path) -> None:
    dst.symlink_to(src.resolve())


def fresh(out: Path, subdirs: list[tuple[str, ...]]) -> None:
    if out.exists():
        shutil.rmtree(out)
    for parts in subdirs:
        (out.joinpath(*parts)).mkdir(parents=True)


def build_yolo_boxlike(src_ds: str, out_name: str) -> None:
    """ultralytics detect/segment: images/{train,val} + labels/{train,val}."""
    src, out = DATA / src_ds, DATA / out_name
    fresh(out, [(sub, sp) for sub in ("images", "labels") for sp in ("train", "val")])

    for sp in ("train", "val"):
        stems = read_split(src, sp)
        n_lbl = 0
        for s in stems:
            link(src / "images" / s, out / "images" / sp / s)
            lbl = src / "labels" / f"{Path(s).stem}.txt"
            if lbl.exists():  # background images legitimately have no label file
                link(lbl, out / "labels" / sp / lbl.name)
                n_lbl += 1
        print(f"  {out_name}/{sp}: {len(stems)} images, {n_lbl} label files")

    (DATA / f"{out_name}.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(out),
                "train": "images/train",
                "val": "images/val",
                "names": THING_CLASSES,
            },
            sort_keys=False,
        )
    )


def build_yolo_sem() -> None:
    """ultralytics task=semantic: images/{train,val} + masks/{train,val}, mask px = class id."""
    src, out = DATA / "dataset_sem", DATA / "yolo_sem"
    fresh(out, [(sub, sp) for sub in ("images", "masks") for sp in ("train", "val")])

    for sp in ("train", "val"):
        stems = read_split(src, sp)
        for s in stems:
            link(src / "images" / s, out / "images" / sp / s)
            link(src / "labels" / f"{Path(s).stem}.png", out / "masks" / sp / f"{Path(s).stem}.png")
        print(f"  yolo_sem/{sp}: {len(stems)} images + masks")

    (DATA / "yolo_sem.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(out),
                "train": "images/train",
                "val": "images/val",
                "masks_dir": "masks",
                "names": SEM_CLASSES,
            },
            sort_keys=False,
        )
    )


def build_rfdetr(src_ds: str, out_name: str) -> None:
    """rfdetr autodetects YOLO format: {train,valid}/{images,labels} + data.yaml at root."""
    src, out = DATA / src_ds, DATA / out_name
    fresh(out, [(sp, sub) for sp in ("train", "valid") for sub in ("images", "labels")])

    for sp_in, sp_out in (("train", "train"), ("val", "valid")):
        stems = read_split(src, sp_in)
        n_lbl = 0
        for s in stems:
            link(src / "images" / s, out / sp_out / "images" / s)
            lbl = src / "labels" / f"{Path(s).stem}.txt"
            if lbl.exists():
                link(lbl, out / sp_out / "labels" / lbl.name)
                n_lbl += 1
        print(f"  {out_name}/{sp_out}: {len(stems)} images, {n_lbl} label files")

    (out / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "train": "train/images",
                "val": "valid/images",
                "nc": len(THING_CLASSES),
                "names": [THING_CLASSES[i] for i in range(len(THING_CLASSES))],
            },
            sort_keys=False,
        )
    )


def build_dataset_sem(raw: Path | None) -> None:
    """D-FINE sem_seg: images/ (symlinks to leftImg8bit) + labels/*.png (trainIds, 255 = ignore)
    + train.csv/val.csv, the official 2975/500 split."""
    import cv2
    import numpy as np

    if raw is None:
        raise SystemExit("dataset_sem needs --raw /path/to/cityscapes_raw (leftImg8bit/, gtFine/)")
    out = DATA / "dataset_sem"
    if out.exists():  # every other layout links into it: never rebuild it silently
        raise SystemExit(f"{out} already exists; delete it first to rebuild")
    lut = np.full(256, 255, np.uint8)
    for label_id, train_id in LABEL_TO_TRAIN.items():
        lut[label_id] = train_id
    (out / "images").mkdir(parents=True)
    (out / "labels").mkdir()
    for sp in ("train", "val"):
        imgs = sorted((raw / "leftImg8bit" / sp).glob("*/*_leftImg8bit.png"), key=lambda p: p.name)
        if not imgs:
            raise SystemExit(
                f"no images under {raw / 'leftImg8bit' / sp}: unzip "
                f"leftImg8bit_trainvaltest.zip and gtFine_trainvaltest.zip into {raw}"
            )
        names = []
        for img in imgs:
            stem = img.name.removesuffix("_leftImg8bit.png")
            gt = raw / "gtFine" / sp / img.parent.name / f"{stem}_gtFine_labelIds.png"
            if not gt.exists():
                raise SystemExit(f"missing ground truth {gt}")
            link(img, out / "images" / f"{stem}.png")
            labels = lut[cv2.imread(str(gt), cv2.IMREAD_GRAYSCALE)]
            cv2.imwrite(str(out / "labels" / f"{stem}.png"), labels)
            names.append(f"{stem}.png")
        (out / f"{sp}.csv").write_text("\n".join(names) + "\n")
        print(f"  dataset_sem/{sp}: {len(names)} images + trainId labels")


BUILDERS = {
    "yolo_det": lambda: build_yolo_boxlike("dataset_det", "yolo_det"),
    "yolo_seg": lambda: build_yolo_boxlike("dataset_seg", "yolo_seg"),
    "yolo_sem": build_yolo_sem,
    "rfdetr_det": lambda: build_rfdetr("dataset_det", "rfdetr_det"),
    "rfdetr_seg": lambda: build_rfdetr("dataset_seg", "rfdetr_seg"),
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--only",
        nargs="*",
        default=list(BUILDERS),
        choices=["dataset_sem", *BUILDERS],
        help="default: every layout but dataset_sem",
    )
    ap.add_argument(
        "--raw",
        type=Path,
        default=None,
        help="unzipped Cityscapes (leftImg8bit/, gtFine/), for dataset_sem",
    )
    args = ap.parse_args()
    for name in args.only:
        print(f"== {name} ==")
        build_dataset_sem(args.raw) if name == "dataset_sem" else BUILDERS[name]()
    print("\ndone")
