"""
Experiment 4: Stability Analysis + Timing Analysis
Uses saved predictions and timing data from Experiments 1, 2, 3.

Stability:
  For each strategy × dataset (Exp 1, fixed 10K budget):
    - Stack predictions across 4 seeds → (4, n_test, n_classes)
    - Compute std per test point → mean over points → one scalar
  High std = less stable across seeds.

Timing:
  From all three experiment CSVs:
    - total_time = sampling_time + inference_time
    - Stacked bar: sampling vs inference by strategy (Exp 1)
    - Time vs AUC trade-off table (Exp 1)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from configs.config import EXCLUDED_FROM_MAIN

RESULTS_DIR  = Path(__file__).parent.parent / "results"
FIGURES_DIR  = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

STRATEGY_ORDER = ["random", "stratified", "coreset", "prototype", "stratified_coreset"]
STRATEGY_LABELS = {
    "random":             "Random",
    "stratified":         "Stratified",
    "coreset":            "k-Center",
    "prototype":          "Prototype (NE)",
    "stratified_coreset": "Per-Class k-Center",
}
STRATEGY_COLORS = {
    "random":             "#1f77b4",
    "stratified":         "#ff7f0e",
    "coreset":            "#2ca02c",
    "prototype":          "#d62728",
    "stratified_coreset": "#9467bd",
}

DATASET_ORDER = [
    "credit-g", "phoneme", "pendigits",
    "mozilla4", "nomao", "bank-marketing", "adult",
    "volkert", "connect-4", "jannis",
    "numerai28.6", "higgs", "MiniBooNE", "covertype",
]

# Below-budget datasets — pool <= MAX_CONTEXT (10K). On these, every sampler
# returns the entire pool, so per-strategy values are structurally identical
# and must be excluded from cross-strategy aggregates (means, Pareto, timing).
# See the parallel informative_subset() helper in analyze_classification_metrics.py.
BELOW_BUDGET = ["credit-g", "phoneme", "pendigits"]

SEEDS = [1, 2, 3, 4]


# ── Stability analysis ────────────────────────────────────────────────────────

def compute_stability():
    """
    Load Experiment 1 predictions and compute mean prediction std across seeds.
    Returns DataFrame with columns [dataset, strategy, mean_pred_std].
    """
    with open(RESULTS_DIR / "experiment_1_predictions.pkl", "rb") as f:
        preds = pickle.load(f)

    records = []
    for ds in DATASET_ORDER:
        for strat in STRATEGY_ORDER:
            seed_preds = []
            for seed in SEEDS:
                key = (ds, strat, seed)
                if key not in preds:
                    continue
                seed_preds.append(preds[key])   # (n_test, n_classes)

            if len(seed_preds) < 2:
                continue

            arr = np.stack(seed_preds, axis=0)          # (n_seeds, n_test, n_classes)
            # std across seeds at each (test_point, class)
            std_arr = arr.std(axis=0)                   # (n_test, n_classes)
            # For binary use the positive-class column; for multiclass average
            mean_std = std_arr.mean()

            records.append({
                "dataset":       ds,
                "strategy":      strat,
                "mean_pred_std": mean_std,
                "n_seeds":       len(seed_preds),
            })

    return pd.DataFrame(records)


def plot_stability(df):
    strategies = [s for s in STRATEGY_ORDER if s in df["strategy"].unique()]
    datasets   = [d for d in DATASET_ORDER  if d in df["dataset"].unique()]

    x     = np.arange(len(strategies))
    width = 0.7 / len(datasets)

    fig, ax = plt.subplots(figsize=(13, 5))

    cmap   = plt.cm.tab10
    colors = [cmap(i / len(datasets)) for i in range(len(datasets))]

    for i, ds in enumerate(datasets):
        sub  = df[df["dataset"] == ds].set_index("strategy")
        vals = [sub.loc[s, "mean_pred_std"] if s in sub.index else 0 for s in strategies]
        offset = (i - len(datasets) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=ds, color=colors[i], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([STRATEGY_LABELS[s] for s in strategies], fontsize=11)
    ax.set_ylabel("Mean prediction std across seeds", fontsize=11)
    ax.set_title(
        "Experiment 4: Prediction Stability Across Seeds\n"
        "(Experiment 1 predictions, fixed 10K budget)",
        fontsize=13,
    )
    ax.legend(ncol=5, fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "experiment_4_stability_by_dataset.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved: figures/experiment_4_stability_by_dataset.png")

    # ── Aggregated bar chart (n = 11 informative datasets) ─────────────────
    # Below-budget datasets contribute tied stability values across strategies
    # (every strategy returns the entire pool, producing identical predictions
    # and therefore identical std), which would bias the cross-strategy mean.
    df_inf = df[~df["dataset"].isin(BELOW_BUDGET)]
    n_inf  = df_inf["dataset"].nunique()
    agg = df_inf.groupby("strategy")["mean_pred_std"].mean().reindex(strategies)

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    bars = ax2.bar(
        range(len(strategies)),
        agg.values,
        color=[STRATEGY_COLORS[s] for s in strategies],
        edgecolor="white", linewidth=0.8,
    )
    ax2.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
    ax2.set_xticks(range(len(strategies)))
    ax2.set_xticklabels([STRATEGY_LABELS[s] for s in strategies], fontsize=10)
    ax2.set_ylabel(f"Mean prediction std (averaged over n = {n_inf} datasets)",
                   fontsize=10)
    ax2.set_title(
        f"Prediction Stability: Mean Across n = {n_inf} Informative Datasets\n"
        "(lower = more stable; below-budget datasets excluded)",
        fontsize=12,
    )
    ax2.grid(axis="y", alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(FIGURES_DIR / "experiment_4_stability_aggregate.png", dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print("Saved: figures/experiment_4_stability_aggregate.png")


def stability_summary(df):
    print("=" * 80)
    print("EXPERIMENT 4: PREDICTION STABILITY (mean std across seeds)")
    print("=" * 80)

    pivot = df.pivot(index="dataset", columns="strategy", values="mean_pred_std")
    pivot = pivot.loc[
        [d for d in DATASET_ORDER if d in pivot.index],
        [s for s in STRATEGY_ORDER if s in pivot.columns],
    ]
    pivot.columns = [STRATEGY_LABELS[s] for s in pivot.columns]
    print("\nPer-dataset mean prediction std (lower = more stable):")
    print(pivot.round(5).to_string())

    # Aggregate uses informative subset (n=11). Below-budget datasets
    # produce structurally tied stability values (every strategy returns
    # the entire pool, so identical predictions across strategies) and
    # would bias the cross-strategy mean.
    df_inf = df[~df["dataset"].isin(BELOW_BUDGET)]
    n_inf = df_inf["dataset"].nunique()
    agg = df_inf.groupby("strategy")["mean_pred_std"].mean().reindex(
        [s for s in STRATEGY_ORDER if s in df_inf["strategy"].unique()])
    print(f"\nMean over n = {n_inf} informative datasets (below-budget excluded):")
    for strat, val in agg.items():
        print(f"  {STRATEGY_LABELS[strat]:20s}: {val:.5f}")

    df.to_csv(RESULTS_DIR / "experiment_4_stability.csv", index=False)
    pivot.to_csv(RESULTS_DIR / "experiment_4_stability_pivot.csv")
    print("\nSaved: experiment_4_stability.csv, experiment_4_stability_pivot.csv")


# ── Timing analysis ───────────────────────────────────────────────────────────

def timing_analysis():
    exp1 = pd.read_csv(RESULTS_DIR / "experiment_1_results.csv")
    if EXCLUDED_FROM_MAIN:
        exp1 = exp1[~exp1["dataset"].isin(EXCLUDED_FROM_MAIN)].reset_index(drop=True)
    exp1["total_time"] = exp1["sampling_time"] + exp1["inference_time"]

    # Filter to informative subset for the cross-strategy timing aggregate.
    # Below-budget datasets have tiny pools, so their TabPFN runs are much
    # faster than the realistic large-pool case the timing should report.
    # Including them biases the per-strategy mean toward small-context speed
    # that the rest of the thesis aggregates exclude.
    exp1_inf = exp1[~exp1["dataset"].isin(BELOW_BUDGET)].reset_index(drop=True)
    n_inf = exp1_inf["dataset"].nunique()

    print("\n" + "=" * 80)
    print(f"TIMING ANALYSIS — EXPERIMENT 1 (fixed budget, n = {n_inf} informative datasets × 4 seeds)")
    print("=" * 80)

    # Mean times per strategy on the informative subset
    grp = exp1_inf.groupby("strategy")[["sampling_time", "inference_time", "total_time"]].mean()
    grp = grp.loc[[s for s in STRATEGY_ORDER if s in grp.index]]
    grp.index = [STRATEGY_LABELS[s] for s in grp.index]
    print("\nMean per-run times (seconds, informative subset):")
    print(grp.round(2).to_string())
    grp.to_csv(RESULTS_DIR / "experiment_4_timing_exp1.csv")
    print("Saved: experiment_4_timing_exp1.csv")

    # Stacked bar: sampling vs inference (Exp 1) on the informative subset
    strategies = [s for s in STRATEGY_ORDER if s in exp1_inf["strategy"].unique()]
    labels = [STRATEGY_LABELS[s] for s in strategies]
    s_times = [exp1_inf[exp1_inf["strategy"]==s]["sampling_time"].mean() for s in strategies]
    i_times = [exp1_inf[exp1_inf["strategy"]==s]["inference_time"].mean() for s in strategies]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars_s = ax.bar(range(len(strategies)), s_times,
                    label="Sampling", color="#aec7e8", edgecolor="white")
    bars_i = ax.bar(range(len(strategies)), i_times, bottom=s_times,
                    label="Inference", color="#1f77b4", edgecolor="white")

    for j, (s, i) in enumerate(zip(s_times, i_times)):
        total = s + i
        ax.text(j, total + 0.5, f"{total:.1f}s", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Time per run (seconds)", fontsize=11)
    ax.set_title(
        f"Sampling + Inference Time by Strategy\n"
        f"(Experiment 1: mean over n = {n_inf} informative datasets × 4 seeds)",
        fontsize=12,
    )
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "experiment_4_timing_stacked.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved: figures/experiment_4_timing_stacked.png")

    # Time vs AUC trade-off table on the informative subset
    tradeoff = exp1_inf.groupby("strategy").agg(
        mean_auc=("auc", "mean"),
        mean_total_time=("total_time", "mean"),
    ).reset_index()
    tradeoff = tradeoff.loc[tradeoff["strategy"].isin(STRATEGY_ORDER)].copy()
    tradeoff["strategy_label"] = tradeoff["strategy"].map(STRATEGY_LABELS)
    tradeoff = tradeoff.set_index("strategy_label")[["mean_auc", "mean_total_time"]].round(4)
    tradeoff.columns = ["Mean AUC", "Mean Total Time (s)"]

    print(f"\nTime vs AUC trade-off (Experiment 1, n = {n_inf} informative datasets):")
    print(tradeoff.to_string())
    tradeoff.to_csv(RESULTS_DIR / "experiment_4_auc_vs_time.csv")
    print("Saved: experiment_4_auc_vs_time.csv")

    # Scaling: how does inference time grow with budget (Exp 2)?
    exp2 = pd.read_csv(RESULTS_DIR / "experiment_2_results.csv")
    exp2["total_time"] = exp2["sampling_time"] + exp2["inference_time"]

    print("\n" + "=" * 80)
    n_exp2_datasets = exp2["dataset"].nunique()
    print(f"TIMING vs BUDGET FRACTION — EXPERIMENT 2 (mean over {n_exp2_datasets} datasets × 4 seeds)")
    print("=" * 80)
    budget_time = exp2.groupby(["strategy", "budget_fraction"])["total_time"].mean().unstack()
    budget_time = budget_time.loc[[s for s in STRATEGY_ORDER if s in budget_time.index]]
    budget_time.index = [STRATEGY_LABELS[s] for s in budget_time.index]
    print(budget_time.round(1).to_string())
    budget_time.to_csv(RESULTS_DIR / "experiment_4_timing_exp2.csv")
    print("Saved: experiment_4_timing_exp2.csv")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_analysis():
    # Stability
    print("Computing prediction stability...")
    stab_df = compute_stability()
    stability_summary(stab_df)
    plot_stability(stab_df)

    # Timing
    timing_analysis()

    print("\n" + "=" * 80)
    print("All Experiment 4 / Timing outputs saved.")
    print("=" * 80)


if __name__ == "__main__":
    run_analysis()
