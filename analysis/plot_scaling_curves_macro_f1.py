"""
Parallel of experiment_2_scaling_curves.png but using macro F1 instead
of AUC-ROC. Used as a defense-ready backup figure to show that the
budget-widening pattern holds across metrics — and that the absolute
gaps on F1 are much larger than on AUC (since F1 is imbalance-aware).

Output: results/figures/experiment_2_scaling_curves_macro_f1.png
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

# Match Experiment 2's dataset ordering — 11 datasets (the 8 original + 3 added later)
DATASET_ORDER = [
    "higgs", "MiniBooNE", "covertype",
    "bank-marketing", "nomao", "connect-4",
    "jannis", "volkert",
    "mozilla4", "adult", "numerai28.6",
]

STRATEGY_STYLES = {
    "random":             {"label": "Random",             "color": "#1f77b4", "marker": "o"},
    "stratified":         {"label": "Stratified",         "color": "#ff7f0e", "marker": "s"},
    "coreset":            {"label": "k-Center",           "color": "#2ca02c", "marker": "^"},
    "prototype":          {"label": "Prototype (NE)",     "color": "#d62728", "marker": "D"},
    "stratified_coreset": {"label": "Per-Class k-Center", "color": "#9467bd", "marker": "v"},
}

BUDGET_LABELS = {0.10: "10%", 0.25: "25%", 0.50: "50%", 1.00: "100%"}


def _grid_dims(n):
    if n <= 3:
        return 1, n
    if n <= 6:
        return 2, 3
    if n <= 9:
        return 3, 3
    return 3, 4


def main():
    df = pd.read_csv(RESULTS_DIR / "classification_metrics_exp2.csv")
    df = df[df["dataset"].isin(DATASET_ORDER)]

    n_ds = len(DATASET_ORDER)
    n_rows, n_cols = _grid_dims(n_ds)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows),
                             sharex=True)
    axes = axes.flatten()

    fracs = sorted(df["budget_fraction"].unique())

    for ax, ds in zip(axes, DATASET_ORDER):
        ds_data = df[df["dataset"] == ds]
        for strategy, style in STRATEGY_STYLES.items():
            means, stds = [], []
            for frac in fracs:
                sub = ds_data[
                    (ds_data["strategy"] == strategy)
                    & (ds_data["budget_fraction"] == frac)
                ]["macro_f1"]
                means.append(sub.mean())
                stds.append(sub.std())
            ax.errorbar(
                fracs, means, yerr=stds,
                color=style["color"], marker=style["marker"],
                label=style["label"], linewidth=2, markersize=7,
                capsize=4, capthick=1.5,
            )
        ax.set_title(ds, fontsize=13, fontweight="bold")
        ax.set_xticks(fracs)
        ax.set_xticklabels([BUDGET_LABELS[f] for f in fracs])
        ax.set_ylabel("Macro F1")
        ax.grid(True, alpha=0.3)

    for ax in axes[n_ds:]:
        ax.set_visible(False)
    for col in range(n_cols):
        last_row_idx = col + (n_rows - 1) * n_cols
        idx = last_row_idx if last_row_idx < n_ds else last_row_idx - n_cols
        if 0 <= idx < n_ds:
            axes[idx].set_xlabel("Budget (% of 10K)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5,
               fontsize=10, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        "Experiment 2: Subsample Size Scaling Curves — Macro F1\n"
        "(Companion to AUC scaling curves;  absolute gaps are typically larger on F1 "
        "because the metric is imbalance-aware.)",
        fontsize=14, y=1.01,
    )
    fig.tight_layout()
    out = FIGURES_DIR / "experiment_2_scaling_curves_macro_f1.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
