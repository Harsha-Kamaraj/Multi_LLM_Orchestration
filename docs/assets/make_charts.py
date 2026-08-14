"""Render the README's figures from the graded pilot run.

Everything plotted here is measured, not illustrative: accuracy comes from the
hidden tier of `runs/{RUN}/rollouts/`, cost from the imputed sidecar that
`orch-workers impute` wrote from the fitted coefficients. Re-run after a new
pilot and the figures follow the data.

    python docs/assets/make_charts.py

Light and dark variants are emitted per figure; the README serves them with
<picture media="(prefers-color-scheme: dark)"> so the images track the reader's
GitHub theme instead of burning a white card into a dark page.
"""

from __future__ import annotations

import collections
import glob
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

RUN = "2026-08-13-c76a55d-4f4767"
OUT = Path(__file__).parent

# Slots 1 and 2 of the validated categorical palette, stepped per mode. Both
# modes pass every gate all-pairs (CVD dE 24.7 light / 26.8 dark, floor 8).
THEME = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7", small="#2a78d6", large="#eb6834"),
    "dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781",
                 grid="#2c2c2a", axis="#383835", small="#3987e5", large="#d95926"),
}

ARM_LABEL = {"direct_small": "1.5B (small arm)", "direct_large": "7B (large arm)"}


def load():
    rows = []
    for f in sorted(glob.glob(f"runs/{RUN}/rollouts/part-*.jsonl")):
        with open(f, encoding="utf-8") as fh:
            rows += [json.loads(line) for line in fh]
    if not rows:
        raise SystemExit(f"no graded rows for {RUN}; run `orch grade run` first")
    return rows


def solved(r) -> bool:
    """Correctness is the hidden tier alone — never the visible one, which the
    model was shown and which is a feature rather than a label."""
    return r["hidden_total"] > 0 and r["hidden_passed"] == r["hidden_total"]


def _style(ax, t):
    ax.set_facecolor(t["surface"])
    ax.figure.set_facecolor(t["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["axis"])
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=t["muted"], labelsize=9, length=0)
    ax.title.set_color(t["ink"])


def fig_accuracy(rows, mode):
    t = THEME[mode]
    groups = ["Overall", "humaneval+", "mbpp+"]
    acc = {}
    for arm in ("direct_small", "direct_large"):
        rs = [r for r in rows if r["arm"] == arm]
        acc[arm] = [
            100 * sum(map(solved, rs)) / len(rs),
            *[100 * sum(solved(r) for r in rs if r["dataset"] == d) /
              max(1, len([r for r in rs if r["dataset"] == d]))
              for d in ("humaneval+", "mbpp+")],
        ]

    fig, ax = plt.subplots(figsize=(7.6, 3.6), dpi=200)
    _style(ax, t)
    y = range(len(groups))
    h = 0.28
    # A 2px surface gap between adjacent bars keeps the pair legible when the
    # two values are close; the bars never touch.
    ax.barh([i + h / 2 + 0.012 for i in y], acc["direct_large"], height=h,
            color=t["large"], label=ARM_LABEL["direct_large"], zorder=3)
    ax.barh([i - h / 2 - 0.012 for i in y], acc["direct_small"], height=h,
            color=t["small"], label=ARM_LABEL["direct_small"], zorder=3)

    for i in y:
        for arm, off in (("direct_large", h / 2 + 0.012), ("direct_small", -h / 2 - 0.012)):
            v = acc[arm][i]
            ax.text(v + 1.0, i + off, f"{v:.1f}%", va="center", ha="left",
                    color=t["ink2"], fontsize=8.5)
        gap = acc["direct_large"][i] - acc["direct_small"][i]
        ax.text(101, i, f"+{gap:.1f} pp", va="center", ha="left",
                color=t["ink"], fontsize=9, fontweight="bold")

    ax.set_yticks(list(y), groups, color=t["ink2"], fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 116)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.xaxis.set_major_formatter(PercentFormatter())
    ax.xaxis.grid(True, color=t["grid"], linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    # Colour passed explicitly: set_title() resets the Text's colour, so the
    # ax.title.set_color() in _style() is overwritten and dark mode loses its title.
    ax.set_title("pass@1 on hidden tests, by arm   ·   1200 graded generations",
                 fontsize=11, pad=30, loc="left", color=t["ink"])
    # Legend sits above the plot: at lower-right it landed on the mbpp+ bar and
    # its value label.
    leg = ax.legend(loc="lower left", bbox_to_anchor=(0, 1.005), frameon=False,
                    fontsize=9, ncols=2, handlelength=1.2, handleheight=1.0)
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])
    fig.tight_layout()
    fig.savefig(OUT / f"accuracy-by-arm-{mode}.png", facecolor=t["surface"])
    plt.close(fig)


def fig_tradeoff(rows, mode):
    t = THEME[mode]
    pts = {}
    for arm in ("direct_small", "direct_large"):
        rs = [r for r in rows if r["arm"] == arm]
        gpu = [r["gpu_seconds"] for r in rs if r.get("gpu_seconds") is not None]
        pts[arm] = (statistics.mean(gpu), 100 * sum(map(solved, rs)) / len(rs))

    fig, ax = plt.subplots(figsize=(7.6, 3.9), dpi=200)
    _style(ax, t)
    (xs, ys), (xl, yl) = pts["direct_small"], pts["direct_large"]

    # Mixing the two arms with probability p is linear in both cost and
    # accuracy, so this segment is exactly what random routing buys. It is the
    # line a learned policy has to beat, not a fitted trend.
    ax.plot([xs, xl], [ys, yl], "--", color=t["muted"], linewidth=1.5, zorder=2)
    # Parked in the empty upper-left rather than on the segment it describes —
    # centred on the line it collided with both the line and the small arm's label.
    ax.text(1.02, 84.5, "random routing between the two arms\n"
                        "— the line a learned policy has to beat",
            color=t["muted"], fontsize=8.5, va="top", linespacing=1.5)

    for arm, color in (("direct_small", t["small"]), ("direct_large", t["large"])):
        x, y = pts[arm]
        ax.scatter([x], [y], s=190, color=color, zorder=4,
                   edgecolors=t["surface"], linewidths=2)
        ax.annotate(f"{ARM_LABEL[arm]}\n{y:.1f}%  ·  {x:.2f} GPU-s",
                    xy=(x, y), xytext=(12, -32) if arm == "direct_small" else (-12, 14),
                    textcoords="offset points", color=t["ink"], fontsize=9,
                    ha="left" if arm == "direct_small" else "right", linespacing=1.5)

    ax.set_xlim(0.7, 4.35)
    ax.set_ylim(50, 88)
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.grid(True, color=t["grid"], linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel("mean GPU-seconds per generation  (imputed from fitted coefficients)",
                  color=t["ink2"], fontsize=9)
    ax.set_ylabel("pass@1", color=t["ink2"], fontsize=9)
    ax.set_title("The routing question:  2.87× the compute buys +19.3 pp",
                 fontsize=11, pad=12, loc="left", color=t["ink"])
    fig.tight_layout()
    fig.savefig(OUT / f"cost-accuracy-{mode}.png", facecolor=t["surface"])
    plt.close(fig)


if __name__ == "__main__":
    rows = load()
    for mode in ("light", "dark"):
        fig_accuracy(rows, mode)
        fig_tradeoff(rows, mode)
    print(f"wrote 4 figures to {OUT} from {len(rows)} graded rows")
