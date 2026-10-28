"""Draw the learning-curve chart for the README: docs/learning_curve.png.

Usage:  python src/plot_learning_curve.py results/learning_curve_<ts>.json

Lines are the median of the 3 seeds at each size (robust to the one DistilBERT run at
5 per intent that never learned); small dots are the individual seeds. The right-most
points are the full-data test results from Parts 1 and 2, the dashed line the LLM.
"""

import json
import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FULL_PER_INTENT = 9002 / 77
FULL = {"baseline": [0.9062], "distilbert": [0.9198, 0.9273, 0.9169]}
LLM_5SHOT, LLM_ZERO = 0.8864, 0.8110

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES = {"distilbert": ("#2a78d6", "Fine-tuned DistilBERT"),
          "baseline": ("#eb6834", "TF-IDF + logistic regression")}


def main() -> None:
    results = pd.DataFrame(json.loads(Path(sys.argv[1]).read_text())["results"])
    fig, ax = plt.subplots(figsize=(8, 5), dpi=200, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    for name, (color, label) in SERIES.items():
        runs = results[results.model == name]
        xs = sorted(runs.per_intent.unique()) + [FULL_PER_INTENT]
        per_x = [runs[runs.per_intent == k].accuracy.tolist() for k in xs[:-1]] + [FULL[name]]
        medians = [pd.Series(v).median() for v in per_x]
        for x, values in zip(xs, per_x):
            ax.scatter([x] * len(values), values, s=10, color=color, alpha=0.35,
                       linewidths=0, zorder=2)
        ax.plot(xs, medians, color=color, linewidth=2, zorder=3, label=label)
        ax.scatter(xs, medians, s=36, color=color, edgecolors=SURFACE, linewidths=1.5,
                   zorder=4)
        ax.annotate(label.split(" ")[0] if name == "baseline" else "DistilBERT",
                    (xs[-1], medians[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK)

    for y, text in [(LLM_5SHOT, "LLM, 5-shot (88.6%)"), (LLM_ZERO, "LLM, zero-shot (81.1%)")]:
        ax.axhline(y, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
        ax.text(5.2, y + 0.006, text, fontsize=8.5, color=MUTED)

    ax.annotate("1 of 3 DistilBERT runs never learned\n(1.3%, stuck at chance; not shown)",
                (5, 0.68), xytext=(6.2, 0.56), fontsize=8, color=MUTED,
                arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.8})

    ax.set_xscale("log")
    ticks = [5, 10, 25, 50, FULL_PER_INTENT]
    ax.set_xticks(ticks, ["5", "10", "25", "50", "all\n(~117)"])
    ax.minorticks_off()
    ax.set_xlim(4.3, 160)
    ax.set_ylim(0.5, 0.95)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Labelled training messages per intent (log scale)", color=MUTED, fontsize=9)
    ax.set_ylabel("Test accuracy (3,080 messages)", color=MUTED, fontsize=9)
    ax.set_title("How much labelled data does training need to match the LLM?",
                 loc="left", fontsize=11, color=INK, pad=12)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=0)
    ax.legend(loc="lower right", frameon=False, fontsize=8.5, labelcolor=INK)

    out = ROOT / "docs" / "learning_curve.png"
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    print(f"Saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
