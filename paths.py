"""Workspace paths shared by every script. Stdlib only: it is imported from every framework's venv.

$CITYSCAPES_ROOT/            default: the directory that contains this repo
  cityscapes-benchmark/      this repo
  data/                      datasets (prep_datasets.py)
  output/                    trained models, prediction dumps, bench CSVs
$DFINE_SEG_ROOT              D-FINE-seg checkout; default: ../../D-FINE-seg from this repo
"""

import os
from pathlib import Path

RIV = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("CITYSCAPES_ROOT") or RIV.parent).resolve()
DATA = ROOT / "data"
OUTPUT = ROOT / "output"


def dfine_root() -> Path:
    """The D-FINE-seg checkout whose uv env runs the scorer, D-FINE inference and engine timing."""
    p = Path(os.environ.get("DFINE_SEG_ROOT") or RIV.parent.parent / "D-FINE-seg").resolve()
    if not (p / "dfine_seg").is_dir():
        raise SystemExit(
            f"no D-FINE-seg checkout at {p}. Clone https://github.com/ArgoHA/D-FINE-seg and set "
            "DFINE_SEG_ROOT=/path/to/D-FINE-seg"
        )
    return p
