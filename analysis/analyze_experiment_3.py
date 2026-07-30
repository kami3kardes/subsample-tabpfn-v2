"""
Experiment 3 analysis: diversity vs size.

Outputs:
  - Summary table (console + CSV)
  - Diversity vs size figure (10-panel, 2 strategies)
  - Best M table
  - Inference time comparison
  - M=1 validation against Experiment 1
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import wilcoxon

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

STRATEGY_STYLES = {
    "stratified": {"color": "#ff7f0e", "marker": "s", "label": "Stratified", "linestyle": "-"},
    "random":     {"color": "#1f77b4", "marker": "o", "label": "Random",     "linestyle": "--"},
}

DATASET_ORDER = [
    "credit-g", "phoneme", "pendigits",
    "mozilla4", "nomao", "bank-marketing", "adult",
    "volkert", "connect-4", "jannis",
    "numerai28.6", "higgs", "MiniBooNE", "covertype",
]
M_VALUES = [1, 3, 5, 10]


def _grid_dims(n):
    """Pick (rows, cols) for n subplots."""
    if n <= 4:  return 2, 2
    if n <= 6:  return 2, 3
    if n <= 10: return 2, 5
    if n <= 15: return 3, 5
    return ((n + 4) // 5, 5)


def load_results():
    return pd.read_csv(RESULTS_DIR / "experiment_3_results.csv")


# ── 1. Summary table ─────────────────────────────────────────────────────────

def summary_table(df):
    grouped = df.groupby(["dataset", "inner_strategy", "M"])["auc"]
    summary = grouped.agg(["mean", "std"]).reset_index()
    summary["mean_std"] = summary.apply(
        lambda r: f"{r['mean']:.4f} ± {r['std']:.4f}", axis=1
    )

    print("=" * 90)
    print("1. SUMMARY TABLE (mean AUC ± std over seeds)")
    print("=" * 90)
    for inner in ["stratified", "random"]:
        print(f"\n--- Inner strategy: {inner} ---")
        sub = summary[summary["inner_strategy"] == inner]
        pivot = sub.pivot(index="dataset", columns="M", values="mean_std")
        pivot = pivot.loc[[d for d in DATASET_ORDER if d in pivot.index]]
        print(pivot.to_string())

    # Save numeric
    num = grouped.agg(["mean", "std"]).reset_index()
    num.to_csv(RESULTS_DIR / "experiment_3_summary.csv", index=False)
    print(f"\nSaved: experiment_3_summary.csv")


# ── 2. Diversity vs size figure ───────────────────────────────────────────────

def plot_diversity_vs_size(df):
    n_ds = len(DATASET_ORDER)
    n_rows, n_cols = _grid_dims(n_ds)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4 * n_cols, 4 * n_rows),
                             sharey=False)
    axes = axes.flatten()

    for ax, ds in zip(axes, DATASET_ORDER):
        ds_data = df[df["dataset"] == ds]
        for inner, style in STRATEGY_STYLES.items():
            means, stds = [], []
            for M in M_VALUES:
                sub = ds_data[
                    (ds_data["inner_strategy"] == inner) &
                    (ds_data["M"] == M)
                ]["auc"]
                means.append(sub.mean())
                stds.append(sub.std())
            ax.errorbar(
                M_VALUES, means, yerr=stds,
                color=style["color"], marker=style["marker"],
                linestyle=style["linestyle"],
                label=style["label"], linewidth=2, markersize=7,
                capsize=4, capthick=1.5,
            )
        ax.set_title(ds, fontsize=11, fontweight="bold")
        ax.set_xticks(M_VALUES)
        ax.set_xlabel("M (ensemble size)")
        ax.set_ylabel("AUC-ROC")
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for ax in axes[n_ds:]:
        ax.set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2,
               fontsize=12, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(
        "Experiment 3: Fixed Budget — Diversity vs. Size\n"
        "(Total budget fixed at min(10K, pool size))",
        fontsize=14, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "experiment_3_diversity_vs_size.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("\nSaved: figures/experiment_3_diversity_vs_size.png")


# ── 3. Best M table ──────────────────────────────────────────────────────────

def best_m_table(df):
    means = df.groupby(["dataset", "inner_strategy", "M"])["auc"].mean().reset_index()
    best = means.loc[means.groupby(["dataset", "inner_strategy"])["auc"].idxmax()]
    pivot = best.pivot(index="dataset", columns="inner_strategy", values="M")
    pivot = pivot.loc[[d for d in DATASET_ORDER if d in pivot.index]]

    # Also show best AUC value
    best_auc = means.loc[means.groupby(["dataset", "inner_strategy"])["auc"].idxmax()]
    auc_pivot = best_auc.pivot(index="dataset", columns="inner_strategy", values="auc")
    auc_pivot = auc_pivot.loc[[d for d in DATASET_ORDER if d in auc_pivot.index]]

    print("\n" + "=" * 90)
    print("3. BEST M TABLE (M value that maximises mean AUC per dataset × strategy)")
    print("=" * 90)
    print(pivot.to_string())
    print("\nCorresponding best mean AUC:")
    print(auc_pivot.round(4).to_string())

    pivot.to_csv(RESULTS_DIR / "experiment_3_best_m.csv")
    print("Saved: experiment_3_best_m.csv")


# ── 4. Inference time comparison ──────────────────────────────────────────────

def inference_time_table(df):
    time_means = (
        df.groupby(["dataset", "inner_strategy", "M"])["inference_time"]
        .mean()
        .reset_index()
    )
    pivot = time_means.pivot_table(
        index="dataset", columns=["inner_strategy", "M"],
        values="inference_time"
    ).round(1)

    print("\n" + "=" * 90)
    print("4. MEAN TOTAL INFERENCE TIME (seconds) PER RUN")
    print("=" * 90)
    print(pivot.to_string())

    # Simpler view: total inference time per M across all datasets
    totals = df.groupby(["M"])["inference_time"].mean().reset_index()
    print("\nMean inference time across all datasets:")
    for _, row in totals.iterrows():
        print(f"  M={int(row['M'])}: {row['inference_time']:.1f}s")

    pivot.to_csv(RESULTS_DIR / "experiment_3_inference_times.csv")
    print("Saved: experiment_3_inference_times.csv")


# ── 5. M=1 validation vs Experiment 1 ────────────────────────────────────────

def m1_vs_mk_wilcoxon(df):
    """Pairwise Wilcoxon signed-rank tests: M=1 vs each higher M, per
    inner sampler. Per-dataset mean AUC across seeds is the unit of
    analysis (n = 14 paired observations per test). Unlike Experiment 1,
    no below-budget exclusion is needed because at M >= 2 the
    per-member budget (10000/M) is below every training pool size, so
    every dataset is informative.
    """
    means = (df.groupby(["dataset", "inner_strategy", "M"])["auc"]
             .mean().reset_index())
    rows = []
    for inner in sorted(means["inner_strategy"].unique()):
        sub = means[means["inner_strategy"] == inner]
        m1 = sub[sub["M"] == 1].set_index("dataset")["auc"]
        for k in [3, 5, 10]:
            mk = sub[sub["M"] == k].set_index("dataset")["auc"]
            common = m1.index.intersection(mk.index)
            v1, vk = m1[common].values, mk[common].values
            a_wins = int(np.sum(v1 > vk))
            b_wins = int(np.sum(vk > v1))
            ties = int(np.sum(v1 == vk))
            diffs = v1 - vk
            if np.all(diffs == 0):
                stat, p = np.nan, 1.0
            else:
                stat, p = wilcoxon(v1, vk)
            rows.append({
                "inner_strategy": inner,
                "M_a": 1, "M_b": k,
                "a_wins": a_wins, "b_wins": b_wins, "ties": ties,
                "n_datasets": len(v1),
                "statistic": stat,
                "p_value": round(p, 6),
                "significant_0.05": p < 0.05,
            })
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "experiment_3_wilcoxon.csv", index=False)

    print("\n" + "=" * 90)
    print("EXPERIMENT 3 — PAIRWISE WILCOXON: M=1 vs M=k (per-dataset mean AUC)")
    print("=" * 90)
    print(out.to_string(index=False))
    print(f"\nSaved: experiment_3_wilcoxon.csv")


def validate_m1(df):
    print("\n" + "=" * 90)
    print("5. VALIDATION: M=1 results vs Experiment 1")
    print("=" * 90)

    exp1_path = RESULTS_DIR / "experiment_1_results.csv"
    if not exp1_path.exists():
        print("  Experiment 1 results not found.")
        return

    exp1 = pd.read_csv(exp1_path)
    exp3_m1 = df[df["M"] == 1].copy()

    all_match = True
    for inner in ["stratified", "random"]:
        print(f"\n  Inner strategy: {inner}")
        e3 = exp3_m1[exp3_m1["inner_strategy"] == inner]
        e1 = exp1[exp1["strategy"] == inner]

        for ds in DATASET_ORDER:
            e3_ds = e3[e3["dataset"] == ds].sort_values("seed").reset_index(drop=True)
            e1_ds = e1[e1["dataset"] == ds].sort_values("seed").reset_index(drop=True)
            if e3_ds.empty or e1_ds.empty:
                continue
            diffs = np.abs(e3_ds["auc"].values - e1_ds["auc"].values)
            max_diff = diffs.max()
            match = max_diff < 1e-10
            if not match:
                all_match = False
            status = "MATCH" if match else f"DIFF={max_diff:.2e}"
            print(f"    {ds:20s}: {status}")

    print(f"\n  All M=1 match Experiment 1: {all_match}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_analysis():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = load_results()
    print(f"Loaded {len(df)} rows\n")

    summary_table(df)
    plot_diversity_vs_size(df)
    best_m_table(df)
    inference_time_table(df)
    m1_vs_mk_wilcoxon(df)
    validate_m1(df)

    print("\n" + "=" * 90)
    print("All Experiment 3 analysis outputs saved.")
    print("=" * 90)


if __name__ == "__main__":
    run_analysis()
