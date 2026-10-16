"""Native-resolution Cityscapes semantic segmentation bench (PROTOCOL.md §11).

Every model sees the full 1024x2048 frame in one forward pass and is scored at 1024x2048 by one
scorer (score.py -> D-FINE's SemSegValidator). Each framework runs its own pre/postprocess in its
own venv (all TensorRT 10.13.3.9, fp16, batch 1):

  yolo       YOLO26{n..x}-sem          .venv-yolo       infer_yolo.py          (YOLO.predict)
  segformer  SegFormer-B{0..5}         .venv-segformer  infer_segformer.py     (HF processor + post)
  effvit     EfficientViT-Seg B0..L2   .venv-effvit     infer_efficientvit.py  (their demo pre/post)
  mask2former  Mask2Former R50..Swin-L  .venv-mask2former  infer_mask2former.py  (DefaultPredictor)
  dfine      D-FINE-seg (--dfine)      D-FINE uv env    infer_dfine.py         (TRTModel)

Backends:
  trt    the table. e2e latency from the same pass as mIoU, engine-only latency measured afterwards
         in the same session (measure_engine.py).
  torch  the fp16-correctness gate: the same pre/post with the PyTorch model. |trt - torch| must
         be <= 0.003 mIoU (PROTOCOL.md §11.4).
  slide  SegFormer only: mmseg sliding-window torch, the reproduction gate for the paper numbers.

  # rivals (idle GPU):
  uv run --project "$DFINE_SEG_ROOT" python -m native.bench_semseg_native \
      --families yolo segformer effvit mask2former --backends trt torch slide
  # add D-FINE-seg:
  ... bench_semseg_native.py --fresh 0 --families dfine yolo segformer effvit mask2former \
      --dfine D-FINE-S=<run>/model.engine D-FINE-X=<run>/model.engine
  # then: python -m native.make_results_native

YOLO's protocol reproduction is separate: .venv-yolo/bin/python -m native.repro_yolo_val
"""

import argparse
import pickle
import subprocess
from pathlib import Path

import pandas as pd
from tabulate import tabulate

from common.paths import RIV, dfine_root
from common.paths import ROOT as R

DFINE = dfine_root()
DFINE_PY = ["uv", "run", "--project", str(DFINE), "python"]
OUT = R / "output" / "compare_native"
HW = (1024, 2048)
GATE = 0.003  # max |TRT fp16 - torch| mIoU

# family -> sizes, display name, published Cityscapes val mIoU (%) and the protocol behind it
FAMILIES = {
    "yolo": {
        "sizes": ["n", "s", "m", "l", "x"],
        "name": lambda s: f"YOLO26{s}-sem",
        "published": {"n": 78.3, "s": 80.8, "m": 82.0, "l": 82.9, "x": 83.6},
        "protocol": "whole 1024x2048",  # reproduced: repro_yolo_val.py
    },
    "segformer": {
        "sizes": ["b0", "b1", "b2", "b3", "b4", "b5"],
        "name": lambda s: f"SegFormer-{s.upper()}",
        "published": {"b0": 76.2, "b1": 78.5, "b2": 81.0, "b3": 81.7, "b4": 82.3, "b5": 82.4},
        "protocol": "slide 1024x1024/768",
    },
    "effvit": {
        "sizes": ["b0", "b1", "b2", "b3", "l1", "l2"],
        "name": lambda s: f"EfficientViT-{s.upper()}",
        "published": {
            "b0": 75.653,
            "b1": 80.547,
            "b2": 82.073,
            "b3": 83.016,
            "l1": 82.716,
            "l2": 83.228,
        },
        "protocol": "whole 1024x2048",
    },
    "mask2former": {
        "sizes": ["r50", "r101", "swin_t", "swin_s", "swin_b", "swin_l"],
        "name": lambda s: (
            "Mask2Former-"
            + {
                "r50": "R50",
                "r101": "R101",
                "swin_t": "Swin-T",
                "swin_s": "Swin-S",
                "swin_b": "Swin-B",
                "swin_l": "Swin-L",
            }[s]
        ),
        "published": {
            "r50": 79.4,
            "r101": 80.1,
            "swin_t": 82.1,
            "swin_s": 82.6,
            "swin_b": 83.3,
            "swin_l": 83.3,
        },
        "protocol": "whole 1024x2048",  # their Cityscapes test cfg: MIN_SIZE_TEST 1024, no TTA
    },
}


def infer_cmd(m: dict, backend: str, out: Path, limit: int) -> list:
    fam, s, name = m["family"], m["size"], m["name"]
    tb = "torch" if backend == "slide" else backend
    if fam == "yolo":
        cmd = [
            RIV / ".venv-yolo/bin/python",
            "-m",
            "infer.infer_yolo",
            "--weights",
            RIV / f"yolo26{s}-sem.pt",
            "--task",
            "semantic",
            "--imgsz",
            *HW,
            "--backend",
            tb,
            "--engine-dir",
            RIV / "artifacts" / "yolo" / f"yolo26{s}-sem_{HW[0]}x{HW[1]}",
        ]
    elif fam == "segformer":
        mode = "slide" if backend == "slide" else "whole"
        cmd = [
            RIV / ".venv-segformer/bin/python",
            "-m",
            "infer.infer_segformer",
            "--variant",
            s,
            "--imgsz",
            *HW,
            "--backend",
            tb,
            "--mode",
            mode,
        ]
    elif fam == "effvit":
        cmd = [
            RIV / ".venv-effvit/bin/python",
            "-m",
            "infer.infer_efficientvit",
            "--variant",
            s,
            "--imgsz",
            *HW,
            "--backend",
            tb,
        ]
    elif fam == "mask2former":
        cmd = [
            RIV / ".venv-mask2former/bin/python",
            "-m",
            "infer.infer_mask2former",
            "--variant",
            s,
            "--imgsz",
            *HW,
            "--backend",
            tb,
        ]
    elif fam == "dfine":
        cmd = DFINE_PY + ["-m", "infer.infer_dfine", "--engine", m["engine"], "--task", "sem_seg"]
        if m.get("weights"):
            cmd += ["--weights", m["weights"]]
    return cmd + ["--name", name, "--out", out, "--limit", limit]


def sh(cmd):
    print("  $", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=RIV)


def gpu_busy() -> list[str]:
    q = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return [ln for ln in q.stdout.splitlines() if ln.strip()]


def models_for(args) -> list[dict]:
    out = []
    for fam in args.families:
        if fam == "dfine":
            for spec in args.dfine:
                name, engine = spec.split("=", 1)
                pt = Path(engine).with_name("model.pt")
                out.append(
                    {
                        "family": "dfine",
                        "size": name,
                        "name": name,
                        "engine": engine,
                        "weights": str(pt) if pt.exists() else None,
                    }
                )
            continue
        f = FAMILIES[fam]
        for s in args.sizes or f["sizes"]:
            if s in f["sizes"]:
                out.append({"family": fam, "size": s, "name": f["name"](s)})
    return out


def run_backend(models: list[dict], backend: str, args) -> pd.DataFrame:
    if backend == "torch":
        models = [m for m in models if m["family"] != "dfine"]  # D-FINE's gate is its own bench
    if backend == "slide":
        models = [m for m in models if m["family"] == "segformer"]
    if not models:
        return pd.DataFrame()
    print(f"\n{'=' * 70}\n{backend}: {[m['name'] for m in models]}\n{'=' * 70}")

    dirs, blobs = [], {}
    for m in models:
        d = OUT / f"{m['name']}_{backend}"
        # D-FINE always re-runs: a reused dump would silently keep an older engine under the same name
        if args.fresh or m["family"] == "dfine" or not (d / "preds.pkl").exists():
            sh(infer_cmd(m, backend, d, args.limit))
        blob = pickle.load(open(d / "preds.pkl", "rb"))
        if blob.get("input") != f"{HW[0]}x{HW[1]}":  # e.g. a D-FINE engine exported at 640
            raise ValueError(f"{m['name']}: ran at {blob.get('input')}, not {HW[0]}x{HW[1]}")
        dirs.append(d)
        blobs[m["name"]] = blob

    csv = OUT / f"_score_{backend}.csv"
    sh(DFINE_PY + ["-m", "common.score", "--task", "sem_seg", "--out", csv, "--preds", *dirs])
    df = pd.read_csv(csv, index_col=0).rename(columns={"latency": "e2e_ms"})

    fam = {m["name"]: FAMILIES.get(m["family"], {}) for m in models}
    meta = pd.DataFrame(
        {
            m["name"]: {
                "family": m["family"],
                "size": m["size"],
                "backend": backend,
                "input": blobs[m["name"]]["input"],
                "params_m": blobs[m["name"]].get("params_m"),
                "gmacs": blobs[m["name"]].get("gmacs"),
                "published_mIoU": fam[m["name"]].get("published", {}).get(m["size"]),
                "published_protocol": fam[m["name"]].get("protocol"),
            }
            for m in models
        }
    ).T
    df = meta.join(df)

    if backend == "trt":  # engine-only latency, same session, same loop for every engine
        engines = {m["name"]: blobs[m["name"]]["engine"] for m in models}
        ecsv = OUT / "_engine_ms.csv"
        sh(DFINE_PY + ["-m", "common.measure_engine", "--csv", ecsv, *engines.values()])
        e = pd.read_csv(ecsv, index_col=0)
        df["engine_ms"] = pd.Series(
            {n: float(e.loc[str(Path(p)), "engine_ms"]) for n, p in engines.items()}
        )
        df["engine"] = pd.Series(engines)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--families",
        nargs="+",
        default=["yolo", "segformer", "effvit"],
        choices=["dfine", "yolo", "segformer", "effvit", "mask2former"],
    )
    ap.add_argument("--sizes", nargs="*", default=None, help="restrict rival sizes, e.g. x l2 b2")
    ap.add_argument("--dfine", nargs="*", default=[], help="NAME=path/to/model.engine (1024x2048)")
    ap.add_argument("--backends", nargs="+", default=["trt"], choices=["trt", "torch", "slide"])
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N val images")
    ap.add_argument("--fresh", type=int, default=1, help="0 = reuse existing dumps")
    ap.add_argument(
        "--allow-busy-gpu",
        action="store_true",
        help="latency is meaningless on a shared GPU; only for smoke tests",
    )
    args = ap.parse_args()
    if "dfine" in args.families and not args.dfine:
        ap.error("--families dfine needs --dfine NAME=engine")

    busy = gpu_busy()
    if busy and not args.allow_busy_gpu:
        raise SystemExit("GPU is not idle (latency would be wrong):\n  " + "\n  ".join(busy))

    OUT.mkdir(parents=True, exist_ok=True)
    models = models_for(args)
    tag = f"_limit{args.limit}" if args.limit else ""
    results = {}
    for backend in args.backends:
        df = run_backend(models, backend, args)
        if not df.empty:
            df.to_csv(OUT / f"semseg_native_{backend}{tag}.csv")
            results[backend] = df

    if "trt" in results and "torch" in results:  # fp16 correctness gate
        t, p = results["trt"]["mIoU"], results["torch"]["mIoU"]
        gate = pd.DataFrame({"trt": t, "torch": p, "delta": (t - p).round(4)}).dropna()
        gate["pass"] = gate["delta"].abs() <= GATE
        print(f"\n=== TRT fp16 vs torch gate (|delta| <= {GATE}) ===")
        print(tabulate(gate, headers="keys", tablefmt="github"))

    for backend, df in results.items():
        cols = [
            c
            for c in (
                "family",
                "params_m",
                "gmacs",
                "mIoU",
                "pixel_acc",
                "e2e_ms",
                "engine_ms",
                "published_mIoU",
                "n_images",
            )
            if c in df
        ]
        print(f"\n=== {backend} ===")
        print(tabulate(df[cols], headers="keys", tablefmt="github"))
    print(
        f"\nwrote {OUT}/semseg_native_<backend>{tag}.csv  ->  python -m native.make_results_native"
    )
