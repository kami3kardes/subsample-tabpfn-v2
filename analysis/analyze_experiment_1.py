"""
Experiment 1 analysis: summary table, strategy ranking,
pairwise Wilcoxon tests, and win-count matrix.

Outputs saved to results/.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
from scipy.stats import wilcoxon

from configs.config import EXCLUDED_FROM_MAIN

RESULTS_DIR = Path(__file__).parent.parent / "results"

# Display order — matches the column order used in calibration_*_summary.csv
# and classification_*_summary.csv files. Updated 2026-06-07 so every summary
# CSV reads left-to-right the same way and cannot be misread across files.
STRATEGY_ORDER = ["random", "stratified", "coreset",
                  "prototype", "stratified_coreset"]
STRATEGY_LABELS = {
    "random":             "Random",
    "stratified":         "Stratified",
    "coreset":            "k-Center",
    "prototype":          "Prototype (NE)",
    "stratified_coreset": "Per-Class k-Center",
}
DATASET_ORDER = [
    "credit-g", "phoneme", "pendigits",
    "mozilla4", "nomao", "bank-marketing", "adult",
    "volkert", "connect-4", "jannis",
    "numerai28.6", "higgs", "MiniBooNE", "covertype",
]


def load_results(exclude_main: bool = True) -> pd.DataFrame:
    """Load the raw Experiment 1 results CSV.

    When `exclude_main` is True (default), datasets listed in
    `EXCLUDED_FROM_MAIN` are filtered out so that the resulting DataFrame
    feeds the main thesis rankings, Wilcoxon tests, and summary heatmaps.
    The full data (including excluded datasets) remains on disk; the
    `exclude_main=False`.
    """
    df = pd.read_csv(RESULTS_DIR / "experiment_1_results.csv")
    if exclude_main and EXCLUDED_FROM_MAIN:
        df = df[~df["dataset"].isin(EXCLUDED_FROM_MAIN)].reset_index(drop=True)
    return df


def _order_pivot(pivot: pd.DataFrame) -> pd.DataFrame:
    """Reorder rows by DATASET_ORDER and columns by STRATEGY_ORDER, renaming
    columns to display labels."""
    cols = [s for s in STRATEGY_ORDER if s in pivot.columns]
    rows = [d for d in DATASET_ORDER if d in pivot.index]
    pivot = pivot.loc[rows, cols]
    pivot.columns = [STRATEGY_LABELS[s] for s in pivot.columns]
    return pivot


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Mean ± std AUC per strategy per dataset.

    Two-row reporting per dataset/strategy: shows the mean over valid
    (non-NaN) seeds and the failure count separately, so strategies that
    fail catastrophically on some seeds cannot benefit from having those
    seeds silently excluded from the mean.
    """
    # Per-strategy mean over valid seeds + fail count
    def _agg(g):
        valid = g.dropna()
        fails = int(g.isna().sum())
        mean = valid.mean() if len(valid) else np.nan
        std = valid.std() if len(valid) > 1 else 0.0
        if fails > 0:
            return f"{mean:.4f} ± {std:.4f} ({fails} fail)"
        return f"{mean:.4f} ± {std:.4f}"

    summary = (df.groupby(["dataset", "strategy"])["auc"]
                 .apply(_agg).reset_index())
    pivot = summary.pivot(index="dataset", columns="strategy", values="auc")
    return _order_pivot(pivot)


def overall_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mean AUC across all datasets per strategy under three failure-handling
    conventions, plus an explicit failure count.

    Why three conventions: a failed run (NaN AUC) means TabPFN could not
    produce a usable model on that (dataset, strategy, seed). The naive
    convention — mean over valid (non-NaN) seeds only — silently rewards
    strategies that fail catastrophically by excluding their worst cases
    from the average. Two failure-penalising conventions correct for this:

      - mean_auc_fail_as_zero: a failed seed is treated as AUC = 0
        (worst-possible accounting; reflects that no model exists).
      - mean_auc_fail_as_baseline: a failed seed is treated as AUC = 0.5
        (random-classifier accounting; reflects that you could have
        substituted a coin-flip baseline).

    For each convention we report the per-strategy mean of per-dataset means
    (so every dataset weighs equally regardless of how many failed seeds).
    """
    # Per-dataset, per-strategy fail counts (one row = max 4 failures over 4 seeds)
    fail_counts = (df.groupby(["dataset", "strategy"])["auc"]
                     .apply(lambda g: int(g.isna().sum()))
                     .reset_index().rename(columns={"auc": "fail_count"}))

    # Convention 1: mean over valid seeds only (current default)
    ds_means_valid = (df.groupby(["dataset", "strategy"])["auc"]
                        .mean().reset_index())

    # Convention 2: failure as AUC = 0
    df_fz = df.copy()
    df_fz["auc"] = df_fz["auc"].fillna(0.0)
    ds_means_fz = (df_fz.groupby(["dataset", "strategy"])["auc"]
                       .mean().reset_index()
                       .rename(columns={"auc": "auc_fz"}))

    # Convention 3: failure as AUC = 0.5
    df_fb = df.copy()
    df_fb["auc"] = df_fb["auc"].fillna(0.5)
    ds_means_fb = (df_fb.groupby(["dataset", "strategy"])["auc"]
                       .mean().reset_index()
                       .rename(columns={"auc": "auc_fb"}))

    # Aggregate to per-strategy means
    def _strat_mean(per_ds, col):
        return per_ds.groupby("strategy")[col].mean()

    ranking = pd.DataFrame({
        "mean_auc_valid_only": _strat_mean(ds_means_valid, "auc"),
        "mean_auc_fail_as_zero": _strat_mean(ds_means_fz, "auc_fz"),
        "mean_auc_fail_as_baseline": _strat_mean(ds_means_fb, "auc_fb"),
        "total_failures": fail_counts.groupby("strategy")["fail_count"].sum(),
        "n_dataset_strategy_pairs": fail_counts.groupby("strategy").size(),
    }).reset_index()

    # Sort by the most defensible convention (fail-as-zero) descending
    ranking = ranking.sort_values("mean_auc_fail_as_zero", ascending=False)
    return ranking


def failure_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Per-dataset × strategy failure count. Shows where failures concentrate."""
    fail_pivot = (df.groupby(["dataset", "strategy"])["auc"]
                    .apply(lambda g: int(g.isna().sum()))
                    .unstack("strategy"))
    return _order_pivot(fail_pivot)


def pairwise_wilcoxon(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pairwise Wilcoxon signed-rank tests on per-dataset mean AUC,
    restricted to the informative subset of datasets.

    Aggregation:
      Each dataset contributes one paired observation per strategy
      (the mean AUC over the 4 seeds). The dataset is the unit of
      analysis because the four within-dataset seeds share the same
      train/test split and TabPFN forward pass and are therefore not
      independent samples.

    Informative subset:
      Below-budget datasets (pool size <= target_size = MAX_CONTEXT)
      are excluded before the test. On those datasets every sampler
      returns the entire pool (see `samplers/base.py`) and the AUC
      values are identical by construction across strategies. They
      carry no information about strategy choice and would force
      scipy to fall back to an asymptotic test instead of the exact
      test on the informative observations.

    Datasets where either strategy failed on all seeds (mean is NaN)
    are dropped, and the count reported as `n_fail_pairs`.
    """
    from configs.config import MAX_CONTEXT
    strategies = sorted(df["strategy"].unique())
    rows = []

    # Exclude datasets whose pool size is at or below the target budget.
    # `pool_size` is identical across (strategy, seed) for a given dataset,
    # so taking the first row per dataset is safe.
    pool_sizes = df.groupby("dataset")["pool_size"].first()
    informative_datasets = pool_sizes[pool_sizes > MAX_CONTEXT].index.tolist()
    df_inf = df[df["dataset"].isin(informative_datasets)]

    # Collapse to one mean AUC per (dataset, strategy)
    means = df_inf.groupby(["dataset", "strategy"])["auc"].mean().reset_index()

    for a, b in combinations(strategies, 2):
        means_a = (means[means["strategy"] == a]
                   .set_index("dataset")["auc"].sort_index())
        means_b = (means[means["strategy"] == b]
                   .set_index("dataset")["auc"].sort_index())
        common = means_a.index.intersection(means_b.index)
        vals_a = means_a[common].values
        vals_b = means_b[common].values

        n_total = len(vals_a)
        valid_mask = ~(np.isnan(vals_a) | np.isnan(vals_b))
        n_fail_pairs = int(n_total - valid_mask.sum())
        vals_a = vals_a[valid_mask]
        vals_b = vals_b[valid_mask]

        # Dataset-level wins (max possible = 14)
        a_wins = int(np.sum(vals_a > vals_b))
        b_wins = int(np.sum(vals_b > vals_a))
        ties = int(np.sum(vals_a == vals_b))

        # Wilcoxon test (skip if no valid pairs or all differences are zero)
        if len(vals_a) == 0:
            stat, p = np.nan, np.nan
        else:
            diffs = vals_a - vals_b
            if np.all(diffs == 0):
                stat, p = np.nan, 1.0
            else:
                stat, p = wilcoxon(vals_a, vals_b)

        rows.append({
            "strategy_a": a,
            "strategy_b": b,
            "a_wins": a_wins,
            "b_wins": b_wins,
            "ties": ties,
            "n_fail_pairs": n_fail_pairs,
            "statistic": stat,
            "p_value": round(p, 6) if not np.isnan(p) else np.nan,
            "significant_0.05": (p < 0.05) if not np.isnan(p) else False,
        })

    result = pd.DataFrame(rows)

    # Holm-Bonferroni step-down adjustment across the family of pairwise tests.
    # Promised in Chapter 2 (Multiple comparisons) as a supplementary
    # verification of the unadjusted significance verdicts.
    pvals = result["p_value"].to_numpy()
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.zeros(m)
    prev = 0.0
    for rank, idx in enumerate(order):
        adj = min(pvals[idx] * (m - rank), 1.0)
        adj = max(adj, prev)
        adjusted[idx] = adj
        prev = adj
    result["p_value_holm"] = np.round(adjusted, 6)
    result["significant_holm_0.05"] = result["p_value_holm"] < 0.05

    return result


def win_count_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-dataset win count: for each pair of strategies, count how many
    datasets strategy A has higher mean AUC than strategy B.
    """
    ds_means = df.groupby(["dataset", "strategy"])["auc"].mean().reset_index()
    strategies = sorted(ds_means["strategy"].unique())
    matrix = pd.DataFrame(0, index=strategies, columns=strategies)

    datasets = ds_means["dataset"].unique()
    for ds in datasets:
        sub = ds_means[ds_means["dataset"] == ds].set_index("strategy")["auc"]
        for a, b in combinations(strategies, 2):
            if sub[a] > sub[b]:
                matrix.loc[a, b] += 1
            elif sub[b] > sub[a]:
                matrix.loc[b, a] += 1

    return matrix


def run_analysis():
    df = load_results()
    print(f"Loaded {len(df)} rows: {df.dataset.nunique()} datasets × "
          f"{df.strategy.nunique()} strategies × {df.seed.nunique()} seeds\n")

    # 1. Summary table
    st = summary_table(df)
    print("=" * 90)
    print("SUMMARY TABLE (mean AUC ± std over seeds)")
    print("=" * 90)
    print(st.to_string())
    print()

    # 2. Overall ranking under three failure-handling conventions
    rank = overall_ranking(df)
    print("=" * 90)
    print("OVERALL STRATEGY RANKING — three failure-handling conventions")
    print("=" * 90)
    print(f"  {'Strategy':<25} {'Valid only':>11}  {'Fail=0':>9}  {'Fail=0.5':>9}  {'Failures':>9}")
    print(f"  {'-'*25} {'-'*11}  {'-'*9}  {'-'*9}  {'-'*9}")
    for i, row in rank.iterrows():
        print(f"  {row['strategy']:<25} "
              f"{row['mean_auc_valid_only']:>11.4f}  "
              f"{row['mean_auc_fail_as_zero']:>9.4f}  "
              f"{row['mean_auc_fail_as_baseline']:>9.4f}  "
              f"{int(row['total_failures']):>9}")
    print()
    print("  Note: 'Valid only' = mean over non-failed seeds. Strategies that fail")
    print("  they are under this convention because their failed seeds are excluded.")
    print("  'Fail=0' and 'Fail=0.5' explicitly penalise failures.")
    print()

    # 2b. Failure rate matrix
    fr = failure_rates(df)
    print("=" * 90)
    print("FAILURE RATES (# of 4 seeds that failed, per dataset × strategy)")
    print("=" * 90)
    # Only show rows with any failures to keep it compact
    has_failures = (fr > 0).any(axis=1)
    if has_failures.any():
        print(fr[has_failures].to_string())
    else:
        print("  (no failures recorded)")
    print()

    # 3. Pairwise Wilcoxon tests
    wilcox = pairwise_wilcoxon(df)
    print("=" * 90)
    print("PAIRWISE WILCOXON SIGNED-RANK TESTS")
    print("=" * 90)
    print(wilcox.to_string(index=False))
    print()

    sig = wilcox[wilcox["significant_0.05"]]
    print(f"Significant pairs (p < 0.05): {len(sig)} / {len(wilcox)}")
    if len(sig) > 0:
        for _, row in sig.iterrows():
            winner = row["strategy_a"] if row["a_wins"] > row["b_wins"] else row["strategy_b"]
            print(f"  {row['strategy_a']} vs {row['strategy_b']}: "
                  f"p={row['p_value']:.6f}, winner={winner} "
                  f"({row['a_wins']}-{row['b_wins']}-{row['ties']})")
    print()

    # 4. Win count matrix
    wm = win_count_matrix(df)
    print("=" * 90)
    print("PAIRWISE WIN COUNT MATRIX (row beats column on N datasets)")
    print("=" * 90)
    print(wm.to_string())
    print()

    # Save outputs
    st.to_csv(RESULTS_DIR / "experiment_1_summary.csv")
    rank.to_csv(RESULTS_DIR / "experiment_1_ranking.csv", index=False)
    fr.to_csv(RESULTS_DIR / "experiment_1_failure_rates.csv")
    wilcox.to_csv(RESULTS_DIR / "experiment_1_wilcoxon.csv", index=False)
    wm.to_csv(RESULTS_DIR / "experiment_1_win_matrix.csv")
    print(f"All analysis outputs saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    run_analysis()
