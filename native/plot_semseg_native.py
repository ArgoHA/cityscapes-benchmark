"""COCO-style mIoU-vs-latency figure from RESULTS_SEMSEG_NATIVE.csv (or any bench CSV via --csv).

  uv run --project ../../D-FINE-seg python -m native.plot_semseg_native                 # engine ms, fit all
  uv run --project ../../D-FINE-seg python -m native.plot_semseg_native --latency e2e --xmax 25
  uv run --project ../../D-FINE-seg python -m native.plot_semseg_native --scale broken  # 0-xmax | gap | slow
  uv run --project ../../D-FINE-seg python -m native.plot_semseg_native --scale log
  uv run --project ../../D-FINE-seg python -m native.plot_semseg_native --candidates    # plot_candidates/

Linear axis + --xmax: slower models are listed in an "off chart" note, or drawn at the right edge
with their real values when named in --pin.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as ticker  # noqa: E402
import pandas as pd  # noqa: E402

from common.paths import RESULTS, RIV  # noqa: E402

CSV = RESULTS / "RESULTS_SEMSEG_NATIVE.csv"

# family -> (legend, color, marker, default label offset in points). D-FINE-seg orange, YOLO blue
# (its own brand colour); the rest keep dataviz reference categorical slots 3-5.
FAMILIES = {
    "dfine": ("D-FINE-seg", "#eb6834", "o", (-8, 4)),
    "yolo": ("YOLO26-sem", "#2a78d6", "s", (-7, 5)),
    "effvit": ("EfficientViT-Seg", "#1baf7a", "D", (7, -12)),
    "segformer": ("SegFormer", "#eda100", "^", (7, -12)),
    "mask2former": ("Mask2Former", "#e87ba4", "v", (-7, 5)),
}
# Per-model label offsets (points; dx 0 = centred) where families cross. OFFSETS[col] wins over
# OFFSETS["any"]; checked by eye on every --candidates image.
OFFSETS = {
    "any": {
        "D-FINE-seg-S": (8, -12),
        "D-FINE-seg-M": (8, -12),
        "SegFormer-B1": (-7, 5),
        "EfficientViT-L1": (0, 7),
        "EfficientViT-L2": (0, -16),
        "EfficientViT-B3": (7, -4),
        "Mask2Former-R50": (-8, -4),
        "Mask2Former-R101": (-8, 2),
        "Mask2Former-Swin-T": (-8, -10),
        "Mask2Former-Swin-S": (7, -12),
        "Mask2Former-Swin-L": (-4, -15),
    },
    "engine_ms": {"YOLO26x-sem": (7, -3), "EfficientViT-L1": (0, -16)},
    "e2e_ms": {"D-FINE-seg-N": (-8, 4), "YOLO26n-sem": (7, -4), "SegFormer-B5": (7, -4)},
}
LATENCY = {
    "engine": ("engine_ms", "TensorRT engine latency, ms"),
    "e2e": ("e2e_ms", "End-to-end latency (framework pre + engine + post), ms"),
}
TITLE = "Cityscapes val mIoU vs latency"
SUBTITLE = "native 1024×2048 whole image · TensorRT 10.13 fp16 · batch 1 · RTX 5070 Ti"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
HALO = [pe.withStroke(linewidth=3, foreground="white")]


def label(name: str, row) -> str:
    if row["family"] == "mask2former":
        return name.removeprefix("Mask2Former-")
    return str(row["size"]).removeprefix("D-FINE-seg-").upper()


def draw(ax, df: pd.DataFrame, col: str):
    """Every family as a line; points outside the axis limits are clipped, labels too."""
    for fam, (name, color, marker, off) in FAMILIES.items():
        d = df[df["family"] == fam].sort_values(col)
        if d.empty:
            continue
        ours = fam == "dfine"
        ax.plot(
            d[col],
            100 * d["mIoU"],
            color=color,
            marker=marker,
            label=name,
            lw=2.6 if ours else 2,
            ms=9 if ours else 7,
            mec="white",
            mew=1.5,
            solid_joinstyle="round",
            solid_capstyle="round",
            zorder=4 if ours else 3,
        )
        for n, r in d.iterrows():
            ox, oy = OFFSETS[col].get(n) or OFFSETS["any"].get(n) or off
            ax.annotate(
                label(n, r),
                (r[col], 100 * r["mIoU"]),
                xytext=(ox, oy),
                textcoords="offset points",
                fontsize=9,
                color=INK,
                ha="right" if ox < 0 else "left" if ox > 0 else "center",
                fontweight="bold" if ours else "normal",
                zorder=5,
                path_effects=HALO,
            )


def pin_right(ax, df: pd.DataFrame, col: str, names: list[str], xmax: float):
    """Off-chart models drawn at the right edge with an arrow and their real values."""
    for n in names:
        r = df.loc[n]
        color = FAMILIES[r["family"]][1]
        y = 100 * r["mIoU"]
        ax.plot(
            [xmax],
            [y],
            marker=">",
            ms=10,
            color=color,
            mec="white",
            mew=1.5,
            clip_on=False,
            zorder=6,
        )
        ax.annotate(
            f"{n} →\n{y:.1f} mIoU @ {r[col]:.1f} ms",
            (xmax, y),
            xytext=(-2, 9),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=8.5,
            color=INK,
            path_effects=HALO,
            zorder=6,
        )


def off_chart_note(df: pd.DataFrame, col: str) -> str:
    """One line per family: '<family>  <size> <mIoU> @ <ms> · ...'."""
    lines = []
    for fam, (name, *_) in FAMILIES.items():
        d = df[df["family"] == fam].sort_values(col)
        if not d.empty:
            pts = [f"{label(n, r)} {100 * r['mIoU']:.1f} @ {r[col]:.1f}" for n, r in d.iterrows()]
            lines.append(f"{name}:  " + "  ·  ".join(pts))
    return "Off chart (mIoU @ ms)\n" + "\n".join(lines)


def style(ax):
    ax.grid(color=GRID, lw=1)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED)


def plot(
    df: pd.DataFrame,
    latency: str,
    scale: str,
    xmax: float | None,
    pin: list[str],
    broken: tuple[float, float] | None,
    path: Path,
):
    col, xlabel = LATENCY[latency]
    y = 100 * df["mIoU"]
    ylim = (y.min() - 0.6, y.max() + 0.6)
    note = ""
    if scale == "broken":
        lo, hi = broken or (xmax or 25, None)
        hi = hi or df.loc[df[col] > lo, col].min() * 0.95  # right panel starts at the 1st slow one
        fig, (ax, ax2) = plt.subplots(
            1,
            2,
            figsize=(11, 6),
            sharey=True,
            gridspec_kw={"width_ratios": [3, 1.3], "wspace": 0.06},
        )
        for a in (ax, ax2):
            draw(a, df, col)
            style(a)
        ax.set_xlim(0, lo)
        ax2.set_xlim(hi, df[col].max() * 1.06)
        ax2.spines["left"].set_visible(False)
        ax2.tick_params(axis="y", left=False)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        ax2.xaxis.set_major_locator(ticker.MultipleLocator(20 if df[col].max() > 80 else 10))
        for a, x, dx in ((ax, 1, 0.01), (ax2, 0, 0.023)):  # "/" break marks on the x axis
            a.plot(
                [x - dx, x + dx],
                [-0.015, 0.015],
                transform=a.transAxes,
                color=MUTED,
                lw=1.2,
                clip_on=False,
            )
        fig.supxlabel(xlabel, color=INK, fontsize=10, y=0.025)
        shown = set(df["family"])
    else:
        fig, ax = plt.subplots(figsize=(10, 6.4))
        hi = xmax or df[col].max() * (1.15 if scale == "log" else 1.05)
        off = df[(df[col] > hi) & ~df.index.isin(pin)]
        shown = set(df.loc[(df[col] <= hi) | df.index.isin(pin), "family"])
        draw(ax, df, col)
        style(ax)
        if pin:
            pin_right(ax, df, col, pin, hi)
        if not off.empty:
            note = off_chart_note(off, col)
        if scale == "log":
            ax.set_xscale("log")
            ax.set_xlim(df[col].min() * 0.85, hi)
            ax.xaxis.set_major_locator(ticker.FixedLocator([1, 2, 3, 5, 10, 20, 30, 50, 100]))
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
            ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        else:
            ax.set_xlim(0, hi)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(steps=[1, 2, 5, 10]))
        ax.set_xlabel(xlabel + (" (log scale)" if scale == "log" else ""), color=INK)
    ax.set_ylim(*ylim)
    ax.set_ylabel("Cityscapes val mIoU, %", color=INK)
    handles = [h for h in ax.get_lines() if h.get_label() in {FAMILIES[f][0] for f in shown}]
    ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1, 0.04), frameon=False)
    fig.text(0.012, 0.975, TITLE, fontsize=13, fontweight="bold", color=INK, va="top")
    fig.text(0.012, 0.925, SUBTITLE, fontsize=9.5, color=MUTED, va="top")
    if scale == "broken":  # supxlabel is not tight_layout-compatible
        fig.subplots_adjust(left=0.07, right=0.99, bottom=0.11, top=0.88)
    else:
        fig.tight_layout(rect=(0, 0.075 if note else 0, 1, 0.9))
    if note:  # below the x label, aligned with the axes
        fig.text(
            ax.get_position().x0,
            0.012,
            note,
            fontsize=7.5,
            color=MUTED,
            va="bottom",
            linespacing=1.4,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")


# name -> (scale, xmax, pin); --candidates renders each for engine and e2e
CANDIDATES = {
    "a_linear_xmax25": ("linear", 25, []),
    "b_broken_axis": ("broken", 25, []),
    "c_log": ("log", None, []),
    "d_linear_xmax25_swinl": ("linear", 25, ["Mask2Former-Swin-L"]),
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--csv", type=Path, default=CSV)
    ap.add_argument("--latency", choices=list(LATENCY), default="engine")
    ap.add_argument("--scale", choices=["linear", "log", "broken"], default="linear")
    ap.add_argument(
        "--xmax",
        type=float,
        default=None,
        help="ms; linear: slower models go in the off-chart note (default: fit all); "
        "broken: end of the left panel (default 25)",
    )
    ap.add_argument(
        "--pin",
        nargs="*",
        default=[],
        help="off-chart models drawn at the right edge, e.g. Mask2Former-Swin-L",
    )
    ap.add_argument(
        "--broken",
        nargs=2,
        type=float,
        metavar=("LEFT_END", "RIGHT_START"),
        help="broken axis: left panel 0..LEFT_END, right panel RIGHT_START..max",
    )
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--candidates",
        action="store_true",
        help="render every launch candidate (engine + e2e) into plot_candidates/",
    )
    args = ap.parse_args()
    data = pd.read_csv(args.csv, index_col=0)
    if args.candidates:
        for cname, (scale, xmax, pin) in CANDIDATES.items():
            for lat in LATENCY:
                plot(
                    data,
                    lat,
                    scale,
                    xmax,
                    pin,
                    None,
                    RIV / "plot_candidates" / f"{cname}_{lat}.png",
                )
    else:
        suffix = "" if args.scale == "linear" else f"_{args.scale}"
        plot(
            data,
            args.latency,
            args.scale,
            args.xmax,
            args.pin,
            tuple(args.broken) if args.broken else None,
            args.out or RESULTS / f"semseg_native_{args.latency}{suffix}.png",
        )
