"""
Experiment 2 analysis: scaling curves, retention, spread, minimum useful budget.

Generates all tables and figures for the subsample-size scaling experiment.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

STRATEGY_STYLES = {
    "random":             {"color": "#1f77b4", "marker": "o", "label": "Random"},
    "stratified":         {"color": "#ff7f0e", "marker": "s", "label": "Stratified"},
    "coreset":            {"color": "#2ca02c", "marker": "^", "label": "k-Center"},
    "prototype":          {"color": "#d62728", "marker": "D", "label": "Prototype (NE)"},
    "stratified_coreset": {"color": "#9467bd", "marker": "v", "label": "Per-Class k-Center"},
}

BUDGET_LABELS = {0.10: "10%", 0.25: "25%", 0.50: "50%", 1.00: "100%"}
DATASET_ORDER = ["higgs", "MiniBooNE", "covertype", "bank-marketing", "nomao",
                 "connect-4", "jannis", "volkert",
                 "mozilla4", "adult", "numerai28.6"]


def load_results():
    return pd.read_csv(RESULTS_DIR / "experiment_2_results.csv")


# ── 1. Summary table ────────────────────────────────────────────────────────

def summary_table(df):
    grouped = df.groupby(["dataset", "budget_fraction", "strategy"])["auc"]
    summary = grouped.agg(["mean", "std"]).reset_index()
    summary["mean_std"] = summary.apply(
        lambda r: f"{r['mean']:.4f} ± {r['std']:.4f}", axis=1
    )

    print("=" * 100)
    print("1. SUMMARY TABLE (mean AUC ± std over seeds)")
    print("=" * 100)
    for ds in DATASET_ORDER:
        ds_data = summary[summary["dataset"] == ds]
        pivot = ds_data.pivot(
            index="budget_fraction", columns="strategy", values="mean_std"
        )
        print(f"\n--- {ds} ---")
        print(pivot.to_string())

    # Save numeric version
    summary_num = grouped.agg(["mean", "std"]).reset_index()
    summary_num.to_csv(RESULTS_DIR / "experiment_2_summary.csv", index=False)
    print(f"\nSaved: experiment_2_summary.csv")


# ── 2. Scaling curves figure ────────────────────────────────────────────────

def _grid_dims(n):
    """Pick (rows, cols) for n subplots: 2x2 for n<=4, 2x3 for 5-6, 3x3 for 7+."""
    if n <= 4:
        return 2, 2
    if n <= 6:
        return 2, 3
    if n <= 9:
        return 3, 3
    return ((n + 2) // 3, 3)


def plot_scaling_curves(df):
    n_ds = len(DATASET_ORDER)
    n_rows, n_cols = _grid_dims(n_ds)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows),
                             sharex=True)
    axes = axes.flatten()

    fracs = sorted(df["budget_fraction"].unique())
    budgets = [int(10000 * f) for f in fracs]

    for ax, ds in zip(axes, DATASET_ORDER):
        ds_data = df[df["dataset"] == ds]
        for strategy, style in STRATEGY_STYLES.items():
            means, stds = [], []
            for frac in fracs:
                sub = ds_data[
                    (ds_data["strategy"] == strategy)
                    & (ds_data["budget_fraction"] == frac)
                ]["auc"]
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
        ax.set_ylabel("AUC-ROC")
        ax.grid(True, alpha=0.3)

    # Hide unused subplots and put x-label on bottom-visible row of each column
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
    fig.suptitle("Experiment 2: Subsample Size Scaling Curves", fontsize=15, y=1.01)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "experiment_2_scaling_curves.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("\nSaved: figures/experiment_2_scaling_curves.png")


# ── 3. Normalized retention figure ──────────────────────────────────────────

def plot_retention(df):
    # Compute per-strategy per-dataset baseline (100% budget mean AUC)
    baseline = (
        df[df["budget_fraction"] == 1.0]
        .groupby(["dataset", "strategy"])["auc"]
        .mean()
    )

    n_ds = len(DATASET_ORDER)
    n_rows, n_cols = _grid_dims(n_ds)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows),
                             sharex=True)
    axes = axes.flatten()
    fracs = sorted(df["budget_fraction"].unique())

    for ax, ds in zip(axes, DATASET_ORDER):
        ds_data = df[df["dataset"] == ds]
        for strategy, style in STRATEGY_STYLES.items():
            base = baseline.loc[(ds, strategy)]
            pcts, stds_pct = [], []
            for frac in fracs:
                sub = ds_data[
                    (ds_data["strategy"] == strategy)
                    & (ds_data["budget_fraction"] == frac)
                ]["auc"]
                pcts.append(100 * sub.mean() / base)
                stds_pct.append(100 * sub.std() / base)
            ax.errorbar(
                fracs, pcts, yerr=stds_pct,
                color=style["color"], marker=style["marker"],
                label=style["label"], linewidth=2, markersize=7,
                capsize=4, capthick=1.5,
            )
        ax.axhline(100, color="gray", linestyle="--", alpha=0.5)
        ax.axhline(95, color="red", linestyle=":", alpha=0.4, label="_95% threshold")
        ax.set_title(ds, fontsize=13, fontweight="bold")
        ax.set_xticks(fracs)
        ax.set_xticklabels([BUDGET_LABELS[f] for f in fracs])
        ax.set_ylabel("% of full-budget AUC")
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
    fig.suptitle("Experiment 2: Performance Retention (% of 10K AUC)",
                 fontsize=15, y=1.01)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "experiment_2_retention.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved: figures/experiment_2_retention.png")


# ── 4. Performance retention table ──────────────────────────────────────────

def retention_table(df):
    baseline = (
        df[df["budget_fraction"] == 1.0]
        .groupby(["dataset", "strategy"])["auc"]
        .mean()
    )
    at_1k = (
        df[df["budget_fraction"] == 0.10]
        .groupby(["dataset", "strategy"])["auc"]
        .mean()
    )
    retention = (100 * at_1k / baseline).reset_index()
    retention.columns = ["dataset", "strategy", "retention_pct"]
    pivot = retention.pivot(index="dataset", columns="strategy", values="retention_pct")
    pivot = pivot.loc[DATASET_ORDER]

    print("\n" + "=" * 100)
    print("4. PERFORMANCE RETENTION TABLE (AUC at 1K as % of AUC at 10K)")
    print("=" * 100)
    print(pivot.round(2).to_string())

    pivot.round(4).to_csv(RESULTS_DIR / "experiment_2_retention_table.csv")
    print("Saved: experiment_2_retention_table.csv")


# ── 5. Strategy spread table ───────────────────────────────────────────────

def spread_table(df):
    ds_means = df.groupby(["dataset", "budget_fraction", "strategy"])["auc"].mean()
    spread = ds_means.groupby(["dataset", "budget_fraction"]).agg(
        lambda x: x.max() - x.min()
    ).reset_index()
    spread.columns = ["dataset", "budget_fraction", "spread"]
    pivot = spread.pivot(index="budget_fraction", columns="dataset", values="spread")
    pivot = pivot[DATASET_ORDER]

    print("\n" + "=" * 100)
    print("5. STRATEGY SPREAD TABLE (best AUC - worst AUC)")
    print("=" * 100)
    print(pivot.round(4).to_string())

    pivot.round(4).to_csv(RESULTS_DIR / "experiment_2_spread_table.csv")
    print("Saved: experiment_2_spread_table.csv")


# ── 6. Minimum useful budget table ─────────────────────────────────────────

def min_budget_table(df):
    baseline = (
        df[df["budget_fraction"] == 1.0]
        .groupby(["dataset", "strategy"])["auc"]
        .mean()
    )
    fracs = sorted(df["budget_fraction"].unique())
    rows = []

    for ds in DATASET_ORDER:
        for strategy in STRATEGY_STYLES:
            base = baseline.loc[(ds, strategy)]
            threshold = 0.95 * base
            min_frac = None
            for frac in fracs:
                sub = df[
                    (df["dataset"] == ds)
                    & (df["strategy"] == strategy)
                    & (df["budget_fraction"] == frac)
                ]["auc"]
                if sub.mean() >= threshold:
                    min_frac = frac
                    break
            rows.append({
                "dataset": ds,
                "strategy": strategy,
                "min_budget_frac": min_frac,
                "min_budget": int(10000 * min_frac) if min_frac else None,
            })

    result = pd.DataFrame(rows)
    pivot = result.pivot(index="dataset", columns="strategy", values="min_budget_frac")
    pivot = pivot.loc[DATASET_ORDER]

    print("\n" + "=" * 100)
    print("6. MINIMUM USEFUL BUDGET (smallest fraction where AUC >= 95% of full-budget)")
    print("=" * 100)
    # Format as percentage strings
    display = pivot.map(lambda x: f"{x:.0%}" if pd.notna(x) else ">100%")
    print(display.to_string())

    pivot.to_csv(RESULTS_DIR / "experiment_2_min_budget_table.csv")
    print("Saved: experiment_2_min_budget_table.csv")


# ── 7. Validation check ────────────────────────────────────────────────────

def validation_check(df):
    print("\n" + "=" * 100)
    print("7. VALIDATION: 100% budget vs Experiment 1")
    print("=" * 100)

    exp1_path = RESULTS_DIR / "experiment_1_results.csv"
    if not exp1_path.exists():
        print("  Experiment 1 results not found, skipping validation.")
        return

    exp1 = pd.read_csv(exp1_path)
    exp2_100 = df[df["budget_fraction"] == 1.0]

    all_match = True
    for ds in DATASET_ORDER:
        e1 = exp1[exp1["dataset"] == ds].sort_values(["strategy", "seed"]).reset_index(drop=True)
        e2 = exp2_100[exp2_100["dataset"] == ds].sort_values(["strategy", "seed"]).reset_index(drop=True)

        diffs = np.abs(e1["auc"].values - e2["auc"].values)
        max_diff = diffs.max()
        n_mismatch = (diffs > 0.001).sum()

        status = "MATCH" if n_mismatch == 0 else "MISMATCH"
        print(f"  {ds:20s}: {status} (max diff: {max_diff:.2e}, mismatches > 0.001: {n_mismatch})")
        if n_mismatch > 0:
            all_match = False

    print(f"\n  All match: {all_match}")


# ── Main ────────────────────────────────────────────────────────────────────

def run_analysis():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = load_results()
    print(f"Loaded {len(df)} rows\n")

    summary_table(df)
    plot_scaling_curves(df)
    plot_retention(df)
    retention_table(df)
    spread_table(df)
    min_budget_table(df)
    validation_check(df)

    print("\n" + "=" * 100)
    print("All Experiment 2 analysis outputs saved.")
    print("=" * 100)


if __name__ == "__main__":
    run_analysis()
