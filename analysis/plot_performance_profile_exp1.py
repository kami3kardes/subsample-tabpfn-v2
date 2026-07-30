"""
Performance profile (Dolan–Moré style) for Experiment 1 AUC.

For each strategy, plots the empirical CDF of its per-dataset
AUC GAP from the best strategy on that dataset. Curves
hugging the left side (small gap) and rising quickly to 1
indicate a strategy that's consistently close to best;
long right tails indicate a strategy that's often far from
best.

Output: results/figures/exp1_performance_profile.png
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

STRAT_ORDER  = ["random", "stratified", "coreset", "prototype", "stratified_coreset"]
STRAT_LABELS = ["Random", "Stratified", "k-Center", "Prototype (NE)", "Per-Class k-Center"]
COLORS = {
    "random":             "#1f77b4",
    "stratified":         "#ff7f0e",
    "coreset":            "#2ca02c",
    "prototype":          "#d62728",
    "stratified_coreset": "#9467bd",
}
MARKERS = {
    "random":             "o",
    "stratified":         "s",
    "coreset":            "^",
    "prototype":          "D",
    "stratified_coreset": "v",
}
EXCLUDED = ["credit-g", "phoneme", "pendigits"]


def main():
    df = pd.read_csv(RESULTS_DIR / "experiment_1_results.csv")

    # Per (dataset, strategy) mean AUC over seeds
    means = df.groupby(["dataset", "strategy"])["auc"].mean().unstack()
    means = means[~means.index.isin(EXCLUDED)]
    means = means[STRAT_ORDER]
    print(f"Datasets retained ({len(means)}): {list(means.index)}")
    print()
    print("Mean AUC per (dataset, strategy):")
    print(means.round(4).to_string())
    print()

    # Per-dataset BEST AUC
    best = means.max(axis=1)
    # Per-strategy GAP from best on each dataset
    gaps = (-means).add(best, axis=0)   # gap = best - strategy

    print("Per-strategy GAP from best (best - strategy):")
    print(gaps.round(4).to_string())
    print()

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6.5))

    for s, label in zip(STRAT_ORDER, STRAT_LABELS):
        sorted_gaps = np.sort(gaps[s].values)
        n = len(sorted_gaps)
        # Empirical CDF: include a starting point at (0, 0) for left edge
        x = np.concatenate(([0.0], sorted_gaps))
        y = np.concatenate(([0.0], np.arange(1, n + 1) / n))
        ax.step(x, y, where="post",
                label=f"{label}",
                color=COLORS[s], linewidth=2.5)
        # Markers at each empirical observation
        ax.scatter(sorted_gaps, np.arange(1, n + 1) / n,
                   marker=MARKERS[s], color=COLORS[s],
                   s=55, edgecolor="black", linewidth=0.5,
                   zorder=3)
        # Annotation: fraction of datasets where this strategy is best
        frac_best = (gaps[s] == 0).sum() / n
        print(f"  {label:25s} — best on {(gaps[s] == 0).sum()}/{n} datasets "
              f"({100*frac_best:.0f}%)")

    ax.axvline(0, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.text(0.0005, 0.05, "x = 0:  strategy is best",
            fontsize=10, color="gray", rotation=0, va="bottom")

    ax.set_xlabel(
        "AUC gap from best strategy on each dataset  (best AUC − strategy AUC)",
        fontsize=12,
    )
    ax.set_ylabel(
        f"Fraction of datasets (n = {len(means)}) at or below the gap",
        fontsize=12,
    )
    ax.set_title(
        "Experiment 1 — Performance Profile (AUC)\n"
        "Empirical CDF of per-dataset AUC gap from best strategy.   "
        "Steeper rise / hug the left edge = consistently close to best.",
        fontsize=12, pad=12,
    )
    ax.legend(loc="lower right", fontsize=11, framealpha=0.92)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(-0.002, gaps.values.max() * 1.05)

    fig.tight_layout()
    out = FIGURES_DIR / "exp1_performance_profile.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
