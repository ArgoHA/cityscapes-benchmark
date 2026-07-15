"""Build YOLO26 / RF-DETR dataset layouts from the existing D-FINE cityscapes datasets.

Each framework gets the data in ITS OWN native layout, at ORIGINAL resolution (2048x1024), and
does its own preprocessing (YOLO letterboxes, RF-DETR square-resizes, D-FINE square-resizes).
We are comparing architecture + framework end-to-end, so no geometry is normalized here.
Images are symlinked - nothing is re-encoded or resized on disk.

Sources (read-only):
  data/dataset_det   images/ + labels/*.txt (bbox)      + train.csv/val.csv
  data/dataset_seg   images/ + labels/*.txt (polygons)  + train.csv/val.csv
  data/dataset       images/ + labels/*.png (trainIds)  + train.csv/val.csv

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

ROOT = Path("/home/argo/Desktop/Projects/cityscapes")
DATA = ROOT / "data"

THING_CLASSES = {
    0: "person", 1: "rider", 2: "car", 3: "truck",
    4: "bus", 5: "train", 6: "motorcycle", 7: "bicycle",
}
SEM_CLASSES = {
    0: "road", 1: "sidewalk", 2: "building", 3: "wall", 4: "fence", 5: "pole",
    6: "traffic-light", 7: "traffic-sign", 8: "vegetation", 9: "terrain", 10: "sky",
    11: "person", 12: "rider", 13: "car", 14: "truck", 15: "bus", 16: "train",
    17: "motorcycle", 18: "bicycle",
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
            {"path": str(out), "train": "images/train", "val": "images/val",
             "names": THING_CLASSES},
            sort_keys=False,
        )
    )


def build_yolo_sem() -> None:
    """ultralytics task=semantic: images/{train,val} + masks/{train,val}, mask px = class id."""
    src, out = DATA / "dataset", DATA / "yolo_sem"
    fresh(out, [(sub, sp) for sub in ("images", "masks") for sp in ("train", "val")])

    for sp in ("train", "val"):
        stems = read_split(src, sp)
        for s in stems:
            link(src / "images" / s, out / "images" / sp / s)
            link(src / "labels" / f"{Path(s).stem}.png", out / "masks" / sp / f"{Path(s).stem}.png")
        print(f"  yolo_sem/{sp}: {len(stems)} images + masks")

    (DATA / "yolo_sem.yaml").write_text(
        yaml.safe_dump(
            {"path": str(out), "train": "images/train", "val": "images/val",
             "masks_dir": "masks", "names": SEM_CLASSES},
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
            {"train": "train/images", "val": "valid/images", "nc": len(THING_CLASSES),
             "names": [THING_CLASSES[i] for i in range(len(THING_CLASSES))]},
            sort_keys=False,
        )
    )


BUILDERS = {
    "yolo_det": lambda: build_yolo_boxlike("dataset_det", "yolo_det"),
    "yolo_seg": lambda: build_yolo_boxlike("dataset_seg", "yolo_seg"),
    "yolo_sem": build_yolo_sem,
    "rfdetr_det": lambda: build_rfdetr("dataset_det", "rfdetr_det"),
    "rfdetr_seg": lambda: build_rfdetr("dataset_seg", "rfdetr_seg"),
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=list(BUILDERS))
    args = ap.parse_args()
    for name in args.only:
        print(f"== {name} ==")
        BUILDERS[name]()
    print("\ndone")
