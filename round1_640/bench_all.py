"""Bench every framework at ITS OWN optimal operating point — accuracy AND latency from the SAME run.

Protocol (two stages, per model):

  stage 1  infer at conf=0.05, dump predictions           -- MECHANISM ONLY, never reported.
           score.py sweeps thresholds over that dump to find t*, the conf that maximises F1.
           (The sweep filters stored predictions, so it can never find an optimum BELOW the conf
           they were dumped at; hence the low floor. No mAP is computed anywhere.)

  stage 2  infer again at conf=t*  -> F1 and latency both come out of THIS run.

Why stage 2 exists: conf directly changes postprocess cost (surviving boxes -> NMS work, mask
upsampling to 1024x2048). Reporting F1@0.25 next to latency@0.5 would be measuring two different
operating points and is not a valid benchmark. Latency and accuracy must be the same pass.

sem_seg has no confidence threshold (dense argmax), so it is a single pass.

Engine-only latency (pure TRT execute) is measured in the same session, so both latency columns
come out of one idle-GPU run and nothing is transcribed by hand.

Each framework runs its OWN shipped inference in its OWN venv; a single scorer (D-FINE's Validator,
original-resolution GT) scores all of them. Run on an IDLE GPU.

  uv run --project "$DFINE_SEG_ROOT" python -m round1_640.bench_all [--models dfine yolo]
"""

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd
from tabulate import tabulate

from common.paths import RIV, dfine_root
from common.paths import ROOT as R

DFINE = dfine_root()
C = R / "output" / "compare"
M = R / "output" / "models"
DFINE_PY = ["uv", "run", "--project", str(DFINE), "python"]
SWEEP_CONF = 0.05  # stage-1 floor: mechanism for the sweep, never reported

# task -> [(run dir, display name), ...]. sem_seg carries S *and* m: m is the size that actually
# matches YOLO26-l/m on latency, and pinning the row to S was under-matching our own model.
DF = {
    "detect": [("det_s_2026-07-11", "D-FINE-S")],
    "segment": [("seg_s_2026-07-11", "D-FINE-S")],
    "sem_seg": [("sem_seg_s_2026-07-10", "D-FINE-S"), ("m_2026-07-11", "D-FINE-m")],
}
CFG = {t: M / v[0][0] / "config.yaml" for t, v in DF.items()}  # GT source (same dataset for all)
# task -> [(ultralytics task, size, run dir, display name), ...]
#   sem_seg  : TWO rows. l is the latency-match to D-FINE-S (1.62 vs 1.52 ms engine), m the
#              closer param-match (14.3 M vs 8.0 M; l is 17.9 M). Report both.
#   detect   : TWO rows, batch 5 vs batch 10. Batch 5 was AutoBatch's (broken) pick; we retrained
#              at 10 to remove the "unfairly small batch" caveat and it came out WORSE on
#              ultralytics' own val mAP50 (0.593 vs 0.611) - fewer optimizer steps/epoch at a fixed
#              nbs=64 accumulation. Bench both and report YOLO's BETTER run; never report a rival's
#              weaker config just because we prefer its training knob.
YOLO = {
    "detect": [
        ("detect", "m", "yolo26m_detect_bs5", "YOLO26-m"),
        ("detect", "m", "yolo26m_detect", "YOLO26-m-bs10"),
    ],
    "segment": [("segment", "m", "yolo26m_segment", "YOLO26-m")],
    "sem_seg": [
        ("semantic", "l", "yolo26l_semantic", "YOLO26-l"),
        ("semantic", "m", "yolo26m_semantic", "YOLO26-m"),
    ],
}
RFDETR = {"detect": [("medium", "rfdetr_medium")], "segment": [("seg-medium", "rfdetr_seg_medium")]}


def runs_for(task: str, models: list[str]) -> list[dict]:
    """Every (framework, variant) row to bench for this task."""
    out = []
    for m in models:
        if m == "dfine":
            for run, name in DF.get(task, []):
                out.append(
                    {"kind": "dfine", "name": name, "slug": run, "engine": M / run / "model.engine"}
                )
        elif m == "yolo":
            for ytask, size, run, name in YOLO.get(task, []):
                out.append(
                    {
                        "kind": "yolo",
                        "name": name,
                        "slug": run,
                        "spec": (ytask, run),
                        "engine": M / run / "weights" / "best.engine",
                    }
                )
        elif m == "rfdetr":
            for variant, run in RFDETR.get(task, []):
                out.append(
                    {
                        "kind": "rfdetr",
                        "name": f"RF-DETR-{variant}",
                        "slug": f"rfdetr_{variant.replace('-', '_')}",
                        "spec": (variant, run),
                        "engine": None,
                    }
                )  # rfdetr builds its engine inside the dump dir
    return out


def resolve_engine(run: dict, task: str) -> Path:
    """rfdetr re-exports ONNX + builds its engine under the dump dir, so it only exists post-infer."""
    if run["engine"] is not None:
        return run["engine"]
    hits = sorted((C / f"final_{run['slug']}_{task}" / "onnx").glob("*.engine"))
    if not hits:
        raise FileNotFoundError(f"no engine for {run['name']} — did stage 2 run?")
    return hits[0]


def sh(cmd):
    print("  $", " ".join(str(c) for c in cmd[-6:]), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=RIV)


def infer(run: dict, task: str, conf: float, out: Path):
    if run["kind"] == "dfine":
        sh(
            DFINE_PY
            + [
                "-m",
                "infer.infer_dfine",
                "--engine",
                run["engine"],
                "--task",
                task,
                "--name",
                run["name"],
                "--conf",
                conf,
                "--out",
                out,
            ]
        )
    elif run["kind"] == "yolo":
        ytask, sub = run["spec"]
        sh(
            [
                RIV / ".venv-yolo/bin/python",
                "-m",
                "infer.infer_yolo",
                "--weights",
                M / sub / "weights" / "best.pt",
                "--task",
                ytask,
                "--name",
                run["name"],
                "--conf",
                conf,
                "--out",
                out,
            ]
        )
    elif run["kind"] == "rfdetr":
        variant, sub = run["spec"]
        sh(
            [
                RIV / ".venv-rfdetr/bin/python",
                "-m",
                "infer.infer_rfdetr",
                "--variant",
                variant,
                "--weights",
                M / sub / "checkpoint_best_ema.pth",
                "--name",
                run["name"],
                "--conf",
                conf,
                "--out",
                out,
            ]
        )


def score(task: str, preds: list[Path], sweep: bool, conf: float, out_csv: Path):
    cmd = (
        DFINE_PY
        + [
            "-m",
            "common.score",
            "--task",
            task,
            "--cfg",
            CFG[task],
            "--sweep",
            1 if sweep else 0,
            "--conf",
            conf,
            "--out",
            out_csv,
            "--preds",
        ]
        + list(preds)
    )
    if task == "sem_seg":  # frozen config predates reorg; sem_seg GT now lives in data/dataset_sem
        cmd += ["--data-path", R / "data" / "dataset_sem"]
    sh(cmd)
    return pd.read_csv(out_csv, index_col=0)


def engine_latency(runs: list[dict], task: str) -> tuple[dict, dict]:
    """Pure TRT execute, same loop for every engine — measured in this same idle-GPU session.
    Also returns each engine's input resolution: it is NOT the same across frameworks (RF-DETR
    keeps its own default 576/432, since resolution is how it buys speed), so it must be reported.
    """
    csv = C / f"engine_ms_{task}.csv"
    eng = {r["name"]: resolve_engine(r, task) for r in runs}
    sh(DFINE_PY + ["-m", "common.measure_engine", "--csv", csv] + list(eng.values()))
    df = pd.read_csv(csv, index_col=0)
    return (
        {n: float(df.loc[str(p), "engine_ms"]) for n, p in eng.items()},
        {n: str(df.loc[str(p), "input"]) for n, p in eng.items()},
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["dfine", "yolo"])
    ap.add_argument("--tasks", nargs="+", default=["detect", "segment", "sem_seg"])
    ap.add_argument("--fresh", type=int, default=1, help="ignore cached stage-1 dumps")
    args = ap.parse_args()

    final = {}
    for task in args.tasks:
        runs = runs_for(task, args.models)
        print(f"\n{'=' * 70}\n{task}: {[r['name'] for r in runs]}\n{'=' * 70}")

        if task == "sem_seg":
            # no confidence threshold (dense argmax) -> one pass, latency + mIoU together
            dirs = []
            for r in runs:
                d = C / f"final_{r['slug']}_{task}"
                infer(r, task, 0.5, d)
                dirs.append(d)
            df = score(task, dirs, sweep=False, conf=0.5, out_csv=C / f"{task}_final.csv")
        else:
            # ---- stage 1: low-conf dump -> sweep -> t* (mechanism, not reported) ----
            s1 = []
            for r in runs:
                d = C / f"acc_{r['slug']}_{task}"
                if args.fresh or not (d / "preds.pkl").exists():
                    infer(r, task, SWEEP_CONF, d)
                s1.append(d)
            sweep_df = score(task, s1, sweep=True, conf=0.5, out_csv=C / f"{task}_sweep.csv")
            t_star = {n: float(row["opt_thresh"]) for n, row in sweep_df.iterrows()}
            print("\n  optimal thresholds:", t_star)

            # ---- stage 2: re-infer AT t* -> accuracy and latency from the same run ----
            rows = {}
            for r in runs:
                d = C / f"final_{r['slug']}_{task}"
                t = t_star[r["name"]]
                infer(r, task, t, d)
                # score each model at its OWN t* (score.py filters at --conf) -> one at a time
                m = score(task, [d], sweep=False, conf=t, out_csv=C / f"_{task}_{r['slug']}.csv")
                rec = m.loc[r["name"]].to_dict()
                rec["conf"] = t
                rows[r["name"]] = rec
            df = pd.DataFrame.from_dict(rows, orient="index")

        ms, res = engine_latency(runs, task)
        df["engine_ms"] = pd.Series(ms)
        df["input"] = pd.Series(res)
        df.to_csv(C / f"{task}_final.csv")
        final[task] = df

    print(
        f"\n\n{'#' * 70}\n#  FINAL — accuracy and latency at each model's own optimal conf\n{'#' * 70}"
    )
    for task, df in final.items():
        print(f"\n=== {task} ===")
        print(tabulate(df.round(3), headers="keys", tablefmt="github", showindex=True))
    json.dump(
        {k: v.to_dict() for k, v in final.items()},
        open(C / "final_all.json", "w"),
        indent=2,
        default=str,
    )
