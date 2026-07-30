"""
Experiment 1 summary figures:
  A) Heatmap — rows=datasets, cols=strategies, color=AUC relative to row mean
  B) Grouped bar chart — per-dataset grouped bars, error bars from 4 seeds
  C) Per-dataset strategy ranking table (rank 1-5 per dataset)
  D) Overall win-count table
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

from configs.config import EXCLUDED_FROM_MAIN

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

STRATEGY_ORDER = ["random", "stratified", "coreset", "prototype", "stratified_coreset"]
STRATEGY_LABELS = {
    "random":             "Random",
    "stratified":         "Stratified",
    "coreset":            "k-Center",
    "prototype":          "Prototype\n(NE)",
    "stratified_coreset": "Per-Class\nk-Center",
}
STRATEGY_COLORS = {
    "random":             "#1f77b4",
    "stratified":         "#ff7f0e",
    "coreset":            "#2ca02c",
    "prototype":          "#d62728",
    "stratified_coreset": "#9467bd",
}
# Ordered by training pool size ascending (small → large)
# credit-g(800) → phoneme(4K) → mozilla4(12K) → bank-marketing(36K) →
# adult(39K) → jannis(67K) → numerai28.6(77K) → higgs(78K) →
# MiniBooNE(104K) → covertype(465K)
DATASET_ORDER = [
    "credit-g", "phoneme", "pendigits",
    "mozilla4", "nomao", "bank-marketing", "adult",
    "volkert", "connect-4", "jannis",
    "numerai28.6", "higgs", "MiniBooNE", "covertype",
]


def load_means(df):
    """Return pivot: rows=datasets, cols=strategies, values=mean AUC over seeds."""
    means = df.groupby(["dataset", "strategy"])["auc"].mean().reset_index()
    pivot = means.pivot(index="dataset", columns="strategy", values="auc")
    pivot = pivot.loc[
        [d for d in DATASET_ORDER if d in pivot.index],
        [s for s in STRATEGY_ORDER if s in pivot.columns],
    ]
    return pivot


def load_stds(df):
    stds = df.groupby(["dataset", "strategy"])["auc"].std().reset_index()
    pivot = stds.pivot(index="dataset", columns="strategy", values="auc")
    pivot = pivot.loc[
        [d for d in DATASET_ORDER if d in pivot.index],
        [s for s in STRATEGY_ORDER if s in pivot.columns],
    ]
    return pivot


# ── A. Heatmap ────────────────────────────────────────────────────────────────

def plot_heatmap(means_pivot):
    strategies = list(means_pivot.columns)
    datasets   = list(means_pivot.index)

    # Relative values: subtract per-row mean so colors show within-dataset differences
    row_means  = means_pivot.mean(axis=1)
    relative   = means_pivot.sub(row_means, axis=0)

    # Per-row std for symmetric colour range
    row_std = relative.abs().max(axis=1).max()
    vmax = max(row_std, 0.002)  # floor to avoid degenerate case

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(
        relative.values,
        cmap="RdYlGn",
        vmin=-vmax, vmax=vmax,
        aspect="auto",
    )

    # Annotate with absolute AUC
    for i, ds in enumerate(datasets):
        for j, strat in enumerate(strategies):
            val = means_pivot.loc[ds, strat]
            rel = relative.loc[ds, strat]
            # White text on dark cells, black on light
            brightness = (rel + vmax) / (2 * vmax)
            text_color = "white" if brightness < 0.25 or brightness > 0.75 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=9, color=text_color, fontweight="bold")

    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(
        [STRATEGY_LABELS[s] for s in strategies],
        fontsize=10, ha="center",
    )
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels(datasets, fontsize=10)
    ax.set_xlabel("Sampling strategy", fontsize=11, labelpad=8)
    ax.set_ylabel("Dataset", fontsize=11)
    ax.set_title(
        "Experiment 1: Mean AUC-ROC by Strategy and Dataset\n"
        "(colour = deviation from per-dataset mean; values = absolute AUC)",
        fontsize=12, pad=10,
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Deviation from row mean", fontsize=9)

    fig.tight_layout()
    out = FIGURES_DIR / "experiment_1_heatmap.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: figures/experiment_1_heatmap.png")


# ── B. Grouped bar chart ──────────────────────────────────────────────────────

def plot_grouped_bars(means_pivot, stds_pivot):
    strategies = list(means_pivot.columns)
    datasets   = list(means_pivot.index)

    n_ds    = len(datasets)
    n_strat = len(strategies)
    group_w = 0.8
    bar_w   = group_w / n_strat
    x       = np.arange(n_ds)

    fig, ax = plt.subplots(figsize=(16, 5))

    for j, strat in enumerate(strategies):
        offset = (j - n_strat / 2 + 0.5) * bar_w
        vals   = means_pivot[strat].values
        errs   = stds_pivot[strat].values
        ax.bar(
            x + offset, vals, bar_w,
            yerr=errs, capsize=3,
            color=STRATEGY_COLORS[strat],
            label=STRATEGY_LABELS[strat].replace("\n", " "),
            alpha=0.88, error_kw={"linewidth": 1, "ecolor": "#444"},
        )

    # Per-dataset y-range: set ylim to [global_min - pad, 1.0]
    all_vals = means_pivot.values.flatten()
    ymin = max(0, all_vals.min() - 0.03)
    ax.set_ylim(ymin, 1.01)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("Mean AUC-ROC (± std over 4 seeds)", fontsize=11)
    ax.set_title(
        "Experiment 1: Strategy Comparison by Dataset",
        fontsize=13,
    )
    ax.legend(ncol=5, fontsize=9, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.3f"))

    fig.tight_layout()
    out = FIGURES_DIR / "experiment_1_bars.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: figures/experiment_1_bars.png")


# ── C. Per-dataset ranking table ─────────────────────────────────────────────

def ranking_table(means_pivot):
    """Rank strategies 1–5 per dataset (1 = best)."""
    ranked = means_pivot.rank(axis=1, ascending=False, method="min").astype(int)
    ranked.columns = [STRATEGY_LABELS[s].replace("\n", " ") for s in ranked.columns]

    print("\n" + "=" * 80)
    print("PER-DATASET STRATEGY RANKING (1 = best)")
    print("=" * 80)
    print(ranked.to_string())

    ranked.to_csv(RESULTS_DIR / "experiment_1_dataset_rankings.csv")
    print("Saved: experiment_1_dataset_rankings.csv")
    return ranked


# ── D. Win-count summary ──────────────────────────────────────────────────────

def win_count_summary(means_pivot):
    """
    For each strategy: count datasets where it ranks 1st (outright win)
    and datasets where it ties for 1st.
    """
    strategies = list(means_pivot.columns)
    rows = []
    for strat in strategies:
        col = means_pivot[strat]
        row_max = means_pivot.max(axis=1)
        wins = int((col == row_max).sum())
        mean_rank = means_pivot.rank(axis=1, ascending=False, method="average")[strat].mean()
        mean_auc  = col.mean()
        rows.append({
            "Strategy":    STRATEGY_LABELS[strat].replace("\n", " "),
            "Wins (best AUC on N datasets)": wins,
            "Mean rank":   round(mean_rank, 2),
            "Mean AUC":    round(mean_auc, 4),
        })

    summary = pd.DataFrame(rows).sort_values("Mean AUC", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 80)
    print("OVERALL WIN COUNT & RANKING SUMMARY")
    print("=" * 80)
    print(summary.to_string(index=False))

    summary.to_csv(RESULTS_DIR / "experiment_1_win_summary.csv", index=False)
    print("Saved: experiment_1_win_summary.csv")
    return summary


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    df = pd.read_csv(RESULTS_DIR / "experiment_1_results.csv")
    if EXCLUDED_FROM_MAIN:
        df = df[~df["dataset"].isin(EXCLUDED_FROM_MAIN)].reset_index(drop=True)
    print(f"Loaded {len(df)} rows ({df['dataset'].nunique()} datasets — excluded {EXCLUDED_FROM_MAIN})\n")

    means = load_means(df)
    stds  = load_stds(df)

    # Per-dataset displays use all 14 main datasets (transparency)
    plot_heatmap(means)
    plot_grouped_bars(means, stds)
    ranking_table(means)

    # Aggregate stats (wins, mean rank, mean AUC) use the n=11 informative
    # subset to match the Wilcoxon test and the calibration/timing
    # aggregates. Below-budget datasets contribute structural ties that
    # inflate every strategy's win count equally and shift mean AUC.
    below_budget = ["credit-g", "phoneme", "pendigits"]
    means_inf = means.drop(index=[d for d in below_budget if d in means.index])
    print(f"\nAggregate ranking computed on n = {len(means_inf)} informative datasets "
          f"(below-budget excluded: {below_budget})")
    win_count_summary(means_inf)

    print("\nDone.")


if __name__ == "__main__":
    run()
