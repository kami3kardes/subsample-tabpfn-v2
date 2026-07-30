"""
Variant of the dominance matrix where each cell shows BOTH the count
and the metric abbreviations on which the row significantly beats
the column. Useful for the slide-8 multi-metric refinement.

Output: results/figures/classification_dominance_matrix_annotated.png
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

STRATEGY_ORDER = ["random", "stratified", "coreset", "prototype", "stratified_coreset"]
STRATEGY_LABELS = {
    "random":             "Random",
    "stratified":         "Stratified",
    "coreset":            "k-Center",
    "prototype":          "Prototype\n(NE)",
    "stratified_coreset": "Per-Class\nk-Center",
}

# Short abbreviations for in-cell display
METRIC_ABBR = {
    "accuracy":      "Acc",
    "log_loss":      "LogL",
    "macro_f1":      "F1",
    "balanced_acc":  "BAcc",
    "mcc":           "MCC",
    "pr_auc":        "PR",
    "macro_auc":     "AUC",
    "top2_accuracy": "Top2",
}


def main():
    df = pd.read_csv(RESULTS_DIR / "classification_wilcoxon.csv")

    # Build cell → list of metric abbrevs
    n = len(STRATEGY_ORDER)
    counts = np.full((n, n), np.nan)
    cell_metrics = {(i, j): [] for i in range(n) for j in range(n)}

    for _, row in df.iterrows():
        if not row["significant_0.05"]:
            continue
        a, b = row["strategy_a"], row["strategy_b"]
        winner, loser = (a, b) if row["a_wins"] > row["b_wins"] else (b, a)
        if winner not in STRATEGY_ORDER or loser not in STRATEGY_ORDER:
            continue
        i = STRATEGY_ORDER.index(winner)
        j = STRATEGY_ORDER.index(loser)
        cell_metrics[(i, j)].append(METRIC_ABBR.get(row["metric"], row["metric"][:4]))

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            counts[i, j] = len(cell_metrics[(i, j)])

    # Plot
    fig, ax = plt.subplots(figsize=(11, 8.5))
    cmap = plt.cm.Greens
    im = ax.imshow(counts, cmap=cmap, vmin=0, vmax=8, aspect="auto")

    # Diagonal mask
    for i in range(n):
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1,
                                    facecolor="#dddddd", edgecolor="none"))
        ax.text(i, i, "—", ha="center", va="center",
                fontsize=16, color="#666666")

    # Cell annotations: count + metric list
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            cnt = int(counts[i, j])
            color_count = "white" if cnt >= 6 else "black"
            color_metrics = "white" if cnt >= 6 else "#333333"

            # Big bold count near top-left of cell (compact, leaves room for list)
            ax.text(j - 0.32, i - 0.30, str(cnt),
                    ha="center", va="center",
                    fontsize=22, color=color_count, fontweight="bold")

            # Vertical list of metric abbreviations (stacked, one per line)
            if cnt > 0:
                metrics = cell_metrics[(i, j)]
                # Stack vertically — one metric per line
                text = "\n".join(metrics)
                # Position list to the right of the count, vertically centered
                ax.text(j + 0.08, i, text,
                        ha="center", va="center",
                        fontsize=9, color=color_metrics,
                        fontweight="normal", linespacing=1.25)
            # If cnt == 0, leave cell blank (no annotation)

    labels = [STRATEGY_LABELS[s] for s in STRATEGY_ORDER]
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Loser (column)", fontsize=12, labelpad=10)
    ax.set_ylabel("Winner (row)", fontsize=12, labelpad=10)
    ax.set_title(
        "Pairwise Statistical Dominance Across 8 Classification Metrics\n"
        "Cell = # of metrics where row significantly beats column (Wilcoxon p < 0.05)\n"
        "Abbreviations listed: Acc = accuracy, LogL = log-loss, F1 = macro F1, "
        "BAcc = balanced acc, MCC, PR = PR-AUC, AUC = macro-AUC, Top2 = top-2 acc",
        fontsize=11, pad=14,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02,
                        ticks=list(range(0, 9, 2)))
    cbar.set_label("# metrics with p < 0.05 (row > column)", fontsize=10)

    fig.tight_layout()
    out = FIGURES_DIR / "classification_dominance_matrix_annotated.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
