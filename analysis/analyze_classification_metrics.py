"""
Classification Metrics Analysis — uses existing predictions, no new TabPFN runs.

Metrics computed (from stored predict_proba arrays in Exp 1 + Exp 2 pkl files):
  accuracy        : exact-match rate (argmax == y)
  log_loss        : cross-entropy, eps-clipped (prob quality, harsher than Brier)
  macro_f1        : F1 averaged equally across classes (imbalance-aware)
  balanced_acc    : recall averaged across classes (alternative to macro-F1)
  mcc             : Matthews correlation coefficient (robust to imbalance)
  pr_auc          : average precision (binary: class-1; multiclass: macro mean)
  macro_auc       : OvR AUC averaged equally across classes (vs current weighted OvR)
  top2_accuracy   : true class within top-2 predictions (multiclass only)

Extra outputs:
  per-class AUC for covertype (7-class) and jannis (4-class)
"""

import pickle
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import wilcoxon
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score, log_loss, f1_score, balanced_accuracy_score,
    matthews_corrcoef, average_precision_score, roc_auc_score,
    top_k_accuracy_score,
)

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from configs.config import (
    DATASETS, TEST_SIZE, SPLIT_RANDOM_STATE, TEST_MAX_SIZE,
    EXCLUDED_FROM_MAIN,
)
from preprocessing.data_loader import load_dataset

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

STRATEGY_ORDER = ["random", "stratified", "coreset", "prototype", "stratified_coreset"]
STRATEGY_LABELS_FLAT = {
    "random":             "Random",
    "stratified":         "Stratified",
    "coreset":            "k-Center",
    "prototype":          "Prototype (NE)",
    "stratified_coreset": "Per-Class k-Center",
}
DATASET_ORDER = [
    # Pool ≤ 10K (strategies equivalent)
    "credit-g", "phoneme", "pendigits",
    # Pool > 10K
    "mozilla4", "nomao", "bank-marketing", "adult",
    "volkert", "connect-4", "jannis",
    "numerai28.6", "higgs", "MiniBooNE", "covertype",
]

EPS = 1e-15  # log_loss clipping


def informative_subset(df):
    """Filter df to the n=11 informative datasets (pool > MAX_CONTEXT).

    Below-budget datasets (credit-g, phoneme, pendigits) contribute identical
    values to every strategy by construction (the sampler returns the entire
    pool), so they carry no information about strategy choice and must be
    excluded from cross-strategy aggregate means. Per-dataset displays (rows
    in tables/heatmaps) can still show them for completeness.

    Pool sizes come from the Exp 1 results CSV — canonical source.
    """
    from configs.config import MAX_CONTEXT
    exp1_csv = RESULTS_DIR / "experiment_1_results.csv"
    pool_sizes = (pd.read_csv(exp1_csv)
                  .groupby("dataset")["pool_size"].first())
    informative_ds = pool_sizes[pool_sizes > MAX_CONTEXT].index.tolist()
    return df[df["dataset"].isin(informative_ds)].copy(), informative_ds


# ── Metric helpers ─────────────────────────────────────────────────────────────

def safe_log_loss(probs, y, n_classes):
    """Cross-entropy with eps clip and explicit labels (handles missing classes)."""
    labels = np.arange(n_classes)
    probs_clipped = np.clip(probs, EPS, 1.0 - EPS)
    return log_loss(y, probs_clipped, labels=labels)


def safe_pr_auc(probs, y, n_classes):
    """Average precision. Binary: on class-1 probs. Multiclass: macro mean."""
    if n_classes == 2:
        return average_precision_score(y, probs[:, 1])
    aps = []
    for c in range(n_classes):
        y_c = (y == c).astype(int)
        if y_c.sum() == 0:
            continue
        aps.append(average_precision_score(y_c, probs[:, c]))
    return float(np.mean(aps)) if aps else np.nan


def safe_macro_auc(probs, y, n_classes):
    """OvR macro AUC. Returns nan if a class has 0 test samples."""
    if n_classes == 2:
        return roc_auc_score(y, probs[:, 1])
    present = np.unique(y)
    if len(present) < n_classes:
        # restrict to present classes
        return roc_auc_score(
            y, probs[:, present], multi_class="ovr",
            average="macro", labels=present,
        )
    return roc_auc_score(y, probs, multi_class="ovr", average="macro")


def per_class_auc(probs, y, n_classes):
    """Return list of length n_classes; nan for classes absent in y_test."""
    aucs = []
    for c in range(n_classes):
        y_c = (y == c).astype(int)
        if y_c.sum() == 0 or y_c.sum() == len(y_c):
            aucs.append(np.nan)
        else:
            aucs.append(roc_auc_score(y_c, probs[:, c]))
    return aucs


def safe_top_k(probs, y, n_classes, k=2):
    """Top-k accuracy. Only meaningful for multiclass."""
    if n_classes <= k:
        return np.nan
    return top_k_accuracy_score(y, probs, k=k, labels=np.arange(n_classes))


def compute_all_metrics(probs, y):
    """Compute the full metric bundle for one (probs, y) pair."""
    n_classes = probs.shape[1]
    pred = probs.argmax(axis=1)
    return {
        "accuracy":      accuracy_score(y, pred),
        "log_loss":      safe_log_loss(probs, y, n_classes),
        "macro_f1":      f1_score(y, pred, average="macro", labels=np.arange(n_classes), zero_division=0),
        "balanced_acc":  balanced_accuracy_score(y, pred),
        "mcc":           matthews_corrcoef(y, pred),
        "pr_auc":        safe_pr_auc(probs, y, n_classes),
        "macro_auc":     safe_macro_auc(probs, y, n_classes),
        "top2_accuracy": safe_top_k(probs, y, n_classes, k=2),
        "n_classes":     n_classes,
    }


# ── Dataset loading (reproduce test sets) ─────────────────────────────────────

def load_test_labels():
    """Reproduce the exact y_test used in each experiment."""
    print("Loading datasets to reconstruct test labels...")
    test_labels = {}
    for name, did in DATASETS.items():
        try:
            X, y = load_dataset(did, name)
            _, X_test, _, y_test = train_test_split(
                X, y, test_size=TEST_SIZE, stratify=y,
                random_state=SPLIT_RANDOM_STATE,
            )
            if len(X_test) > TEST_MAX_SIZE:
                sss = StratifiedShuffleSplit(
                    n_splits=1, train_size=TEST_MAX_SIZE,
                    random_state=SPLIT_RANDOM_STATE,
                )
                idx, _ = next(sss.split(X_test, y_test))
                y_test = y_test[idx]
            classes = np.unique(y_test)
            label_map = {c: i for i, c in enumerate(classes)}
            y_test = np.array([label_map[v] for v in y_test])
            test_labels[name] = y_test
            print(f"  {name:20s}: {len(y_test)} samples  classes={len(classes)}")
        except Exception as exc:
            print(f"  [ERROR] {name}: {exc}")
    return test_labels


# ── Per-run metric computation ────────────────────────────────────────────────

def compute_exp1_metrics(preds1, test_labels):
    records = []
    for (ds, strat, seed), probs in preds1.items():
        if ds not in test_labels:
            continue
        y = test_labels[ds]
        m = compute_all_metrics(probs, y)
        records.append({"dataset": ds, "strategy": strat, "seed": seed, **m})
    return pd.DataFrame(records)


def compute_exp2_metrics(preds2, test_labels):
    records = []
    for (ds, strat, frac, seed), probs in preds2.items():
        if ds not in test_labels:
            continue
        y = test_labels[ds]
        m = compute_all_metrics(probs, y)
        records.append({
            "dataset": ds, "strategy": strat,
            "budget_fraction": frac, "seed": seed, **m,
        })
    return pd.DataFrame(records)


# ── Per-class AUC for multiclass datasets ─────────────────────────────────────

def compute_per_class_auc(preds1, test_labels,
                          multiclass_datasets=("covertype", "jannis",
                                                "pendigits",
                                                "connect-4", "volkert")):
    records = []
    for (ds, strat, seed), probs in preds1.items():
        if ds not in multiclass_datasets or ds not in test_labels:
            continue
        y = test_labels[ds]
        aucs = per_class_auc(probs, y, probs.shape[1])
        row = {"dataset": ds, "strategy": strat, "seed": seed}
        for c, auc in enumerate(aucs):
            row[f"class_{c}_auc"] = auc
        records.append(row)
    return pd.DataFrame(records)


# ── Summary tables ─────────────────────────────────────────────────────────────

METRIC_DIRECTION = {
    "accuracy":      "higher",
    "log_loss":      "lower",
    "macro_f1":      "higher",
    "balanced_acc":  "higher",
    "mcc":           "higher",
    "pr_auc":        "higher",
    "macro_auc":     "higher",
    "top2_accuracy": "higher",
}

METRIC_LABELS = {
    "accuracy":      "Accuracy",
    "log_loss":      "Log-Loss",
    "macro_f1":      "Macro-F1",
    "balanced_acc":  "Balanced Accuracy",
    "mcc":           "MCC",
    "pr_auc":        "PR-AUC (Avg Precision)",
    "macro_auc":     "Macro AUC (OvR)",
    "top2_accuracy": "Top-2 Accuracy",
}


def _ordered_pivot(df, metric, agg="mean"):
    p = df.groupby(["dataset", "strategy"])[metric].agg(agg).unstack("strategy")
    p = p.loc[
        [d for d in DATASET_ORDER if d in p.index],
        [s for s in STRATEGY_ORDER if s in p.columns],
    ]
    p.columns = [STRATEGY_LABELS_FLAT[s] for s in p.columns]
    return p


def write_summary_tables(df1, out_prefix="classification"):
    """One CSV per metric: mean ± std per dataset × strategy."""
    for metric in METRIC_DIRECTION:
        if metric not in df1.columns:
            continue
        mean_p = _ordered_pivot(df1, metric, "mean").round(4)
        std_p  = _ordered_pivot(df1, metric, "std").round(4)
        # mean_p may contain NaN (e.g., top2 on binary); preserve as "nan ± nan"
        combined = mean_p.astype(str) + " ± " + std_p.astype(str)
        out = RESULTS_DIR / f"{out_prefix}_{metric}_summary.csv"
        combined.to_csv(out)
    print(f"Saved per-metric summary CSVs: {out_prefix}_<metric>_summary.csv "
          f"({len(METRIC_DIRECTION)} files)")


def print_summary(df1):
    print("\n" + "=" * 96)
    print("CLASSIFICATION METRICS — PER-DATASET MEANS (Exp 1, 14 main datasets × 4 seeds)")
    print("=" * 96)

    for metric, direction in METRIC_DIRECTION.items():
        if metric not in df1.columns:
            continue
        mean_p = _ordered_pivot(df1, metric, "mean")
        arrow = "↑" if direction == "higher" else "↓"
        print(f"\n{METRIC_LABELS[metric]} ({arrow} = {direction} is better)")
        print("-" * 96)
        print(mean_p.round(4).to_string())


def print_metric_highlights(df1):
    """Best and worst strategy per dataset for each metric."""
    print("\n" + "=" * 96)
    print("BEST / WORST STRATEGY PER DATASET (mean over seeds)")
    print("=" * 96)
    for metric in METRIC_DIRECTION:
        if metric not in df1.columns:
            continue
        direction = METRIC_DIRECTION[metric]
        ascending = direction == "lower"
        means = df1.groupby(["dataset", "strategy"])[metric].mean().reset_index()
        print(f"\n  {METRIC_LABELS[metric]}  ({'lower' if ascending else 'higher'} = better)")
        for ds in DATASET_ORDER:
            sub = means[means["dataset"] == ds].dropna(subset=[metric])
            if sub.empty:
                continue
            sub = sub.sort_values(metric, ascending=ascending)
            best  = sub.iloc[0]
            worst = sub.iloc[-1]
            gap = abs(best[metric] - worst[metric])
            print(f"    {ds:18s}  best: {STRATEGY_LABELS_FLAT[best['strategy']]:17s}"
                  f"={best[metric]:.4f}  |  worst: {STRATEGY_LABELS_FLAT[worst['strategy']]:17s}"
                  f"={worst[metric]:.4f}  |  Δ={gap:.4f}")


def overall_strategy_ranking(df1):
    """Mean of each metric across the n=11 informative datasets per strategy.

    Below-budget datasets (credit-g, phoneme, pendigits) contribute structural
    ties and are excluded — see informative_subset() docstring. This matches
    the convention used in the Wilcoxon tests and in the Ch 7 calibration
    aggregates so that all cross-strategy means in the thesis use the same
    dataset family.
    """
    df_inf, informative_ds = informative_subset(df1)
    n_inf = len(informative_ds)

    rows = []
    for metric in METRIC_DIRECTION:
        if metric not in df_inf.columns:
            continue
        per_ds = df_inf.groupby(["dataset", "strategy"])[metric].mean().reset_index()
        avg = per_ds.groupby("strategy")[metric].mean()
        for strat, val in avg.items():
            rows.append({"metric": metric, "strategy": strat, "mean_over_datasets": val})
    rank_df = pd.DataFrame(rows)
    pivot = rank_df.pivot(index="strategy", columns="metric", values="mean_over_datasets")
    pivot = pivot.loc[[s for s in STRATEGY_ORDER if s in pivot.index]]
    cols = [m for m in METRIC_DIRECTION if m in pivot.columns]
    pivot = pivot[cols]

    print("\n" + "=" * 96)
    print(f"OVERALL STRATEGY RANKING (mean over {n_inf} informative datasets, equal weighting)")
    print("=" * 96)
    print(pivot.round(4).to_string())

    pivot.round(4).to_csv(RESULTS_DIR / "classification_overall_ranking.csv")
    print("\nSaved: classification_overall_ranking.csv")
    return pivot


def pairwise_wilcoxon(df, metric, direction):
    """
    Pairwise Wilcoxon signed-rank tests on a single metric, restricted
    to the informative subset of datasets.

    Aggregation:
      Per-dataset mean over the four seeds. Seeds are aggregated
      because the four within-dataset seeds share the same train/test
      split and TabPFN forward pass and are not independent samples.

    Informative subset:
      Below-budget datasets (pool size <= target_size = MAX_CONTEXT)
      are excluded before the test. On those datasets every sampler
      returns the entire pool, so values are identical across
      strategies by construction and carry no information about
      strategy choice.

    NaN observations (e.g. top-2 accuracy on binary datasets) are
    dropped per dataset when computing the mean.

    Winner is the strategy with higher value if direction=='higher',
    lower value if direction=='lower'.
    """
    from configs.config import MAX_CONTEXT
    strategies = [s for s in STRATEGY_ORDER if s in df["strategy"].unique()]
    rows = []

    # Exclude below-budget datasets (pool_size <= target_size); on those
    # datasets every sampler returns the entire pool, so the comparison
    # is structurally tied. Pool sizes come from the Exp 1 results CSV,
    # which is the canonical source for per-dataset pool size.
    exp1_csv = RESULTS_DIR / "experiment_1_results.csv"
    pool_sizes = (pd.read_csv(exp1_csv)
                  .groupby("dataset")["pool_size"].first())
    informative_datasets = pool_sizes[pool_sizes > MAX_CONTEXT].index.tolist()
    df_inf = df[df["dataset"].isin(informative_datasets)]

    # Collapse to one mean per (dataset, strategy); NaN-skipping mean
    # so a binary dataset with NaN top-2 accuracy is still represented.
    means = (df_inf.groupby(["dataset", "strategy"])[metric]
             .mean().reset_index())

    for a, b in combinations(strategies, 2):
        ma = (means[means["strategy"] == a]
              .set_index("dataset")[metric].sort_index())
        mb = (means[means["strategy"] == b]
              .set_index("dataset")[metric].sort_index())
        common = ma.index.intersection(mb.index)
        va = ma[common].values
        vb = mb[common].values
        # Drop datasets where mean is NaN for either (failure-only)
        mask = ~(np.isnan(va) | np.isnan(vb))
        va, vb = va[mask], vb[mask]

        if direction == "lower":
            # winner = lower value
            a_wins = int(np.sum(va < vb))
            b_wins = int(np.sum(vb < va))
        else:
            a_wins = int(np.sum(va > vb))
            b_wins = int(np.sum(vb > va))
        ties = int(np.sum(va == vb))

        diffs = va - vb
        if len(diffs) == 0 or np.all(diffs == 0):
            stat, p = np.nan, 1.0
        else:
            stat, p = wilcoxon(va, vb)

        rows.append({
            "metric":      metric,
            "strategy_a":  a,
            "strategy_b":  b,
            "n_pairs":     len(va),
            "a_wins":      a_wins,
            "b_wins":      b_wins,
            "ties":        ties,
            "statistic":   stat,
            "p_value":     round(p, 6),
            "significant_0.05": bool(p < 0.05),
        })

    return pd.DataFrame(rows)


def all_metrics_wilcoxon(df):
    """Run Wilcoxon for every metric. Returns long-format DataFrame."""
    frames = []
    for metric, direction in METRIC_DIRECTION.items():
        if metric not in df.columns:
            continue
        frames.append(pairwise_wilcoxon(df, metric, direction))
    return pd.concat(frames, ignore_index=True)


def consensus_summary(wilcox_df):
    """
    For each (strategy_a, strategy_b) pair, count how many of the 8 metrics
    have a_wins > b_wins (i.e. A is better) AND p < 0.05.

    Produces a matrix that surfaces robust pairwise dominance.
    """
    pairs = wilcox_df[["strategy_a", "strategy_b"]].drop_duplicates()
    rows = []
    for _, pair in pairs.iterrows():
        a, b = pair["strategy_a"], pair["strategy_b"]
        sub = wilcox_df[
            (wilcox_df["strategy_a"] == a) & (wilcox_df["strategy_b"] == b)
        ]
        n_total = len(sub)
        a_better_sig = int(((sub["a_wins"] > sub["b_wins"]) & sub["significant_0.05"]).sum())
        b_better_sig = int(((sub["b_wins"] > sub["a_wins"]) & sub["significant_0.05"]).sum())
        not_sig      = int((~sub["significant_0.05"]).sum())
        rows.append({
            "strategy_a": a,
            "strategy_b": b,
            "n_metrics":  n_total,
            "a_better_sig": a_better_sig,
            "b_better_sig": b_better_sig,
            "not_significant": not_sig,
            "verdict": (
                f"A dominates ({a_better_sig}/{n_total})" if a_better_sig >= n_total - 1
                else f"B dominates ({b_better_sig}/{n_total})" if b_better_sig >= n_total - 1
                else "mixed"
            ),
        })
    return pd.DataFrame(rows)


def print_wilcoxon_summary(wilcox_df, consensus_df):
    print("\n" + "=" * 110)
    print("WILCOXON SIGNED-RANK TESTS — 8 METRICS, ALL STRATEGY PAIRS")
    print("(n_pairs per metric = 11 informative datasets; per-dataset means after collapsing 4 seeds)")
    print("=" * 110)

    pairs = wilcox_df[["strategy_a", "strategy_b"]].drop_duplicates()
    for _, pair in pairs.iterrows():
        a, b = pair["strategy_a"], pair["strategy_b"]
        sub = wilcox_df[(wilcox_df["strategy_a"] == a) & (wilcox_df["strategy_b"] == b)]
        print(f"\n  {STRATEGY_LABELS_FLAT[a]} vs {STRATEGY_LABELS_FLAT[b]}")
        print(f"    {'Metric':<22} {'A wins':<8} {'B wins':<8} {'Ties':<6} {'p-value':<12} {'sig?'}")
        print(f"    {'-'*22} {'-'*8} {'-'*8} {'-'*6} {'-'*12} {'-'*5}")
        for _, row in sub.iterrows():
            sig = "**" if row["significant_0.05"] else ""
            print(f"    {METRIC_LABELS[row['metric']]:<22} "
                  f"{row['a_wins']:<8} {row['b_wins']:<8} {row['ties']:<6} "
                  f"{row['p_value']:<12.5f} {sig}")

    print("\n" + "=" * 110)
    print("CONSENSUS SUMMARY — How many of the 8 metrics significantly favor A over B?")
    print("=" * 110)
    print(f"\n  {'Pair':<45} {'A sig wins':<14} {'B sig wins':<14} {'Not sig':<12} Verdict")
    print(f"  {'-'*45} {'-'*14} {'-'*14} {'-'*12} {'-'*30}")
    for _, row in consensus_df.iterrows():
        pair_label = (f"{STRATEGY_LABELS_FLAT[row['strategy_a']]} vs "
                      f"{STRATEGY_LABELS_FLAT[row['strategy_b']]}")
        print(f"  {pair_label:<45} {row['a_better_sig']:<14} "
              f"{row['b_better_sig']:<14} {row['not_significant']:<12} {row['verdict']}")


def plot_macro_f1_heatmap(df1):
    """
    Macro-F1 heatmap analogous to experiment_1_heatmap.png (AUC sister figure).
    Same layout: 12 datasets × 5 strategies, RdYlGn deviation-from-row-mean.

    Surfaces failures softened by AUC: bank-marketing Per-Class k-Center,
    covertype Prototype, MiniBooNE Per-Class k-Center.
    """
    pivot = df1.groupby(["dataset", "strategy"])["macro_f1"].mean().unstack("strategy")
    pivot = pivot.loc[
        [d for d in DATASET_ORDER if d in pivot.index],
        [s for s in STRATEGY_ORDER if s in pivot.columns],
    ]
    strategies = list(pivot.columns)
    datasets   = list(pivot.index)

    row_means = pivot.mean(axis=1)
    relative  = pivot.sub(row_means, axis=0)
    vmax = max(relative.abs().max().max(), 0.002)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(relative.values, cmap="RdYlGn",
                   vmin=-vmax, vmax=vmax, aspect="auto")

    strategy_labels_mline = {
        "random":             "Random",
        "stratified":         "Stratified",
        "coreset":            "k-Center",
        "prototype":          "Prototype\n(NE)",
        "stratified_coreset": "Strat.\nk-Center",
    }

    for i, ds in enumerate(datasets):
        for j, strat in enumerate(strategies):
            val = pivot.loc[ds, strat]
            rel = relative.loc[ds, strat]
            brightness = (rel + vmax) / (2 * vmax)
            text_color = "white" if brightness < 0.25 or brightness > 0.75 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=9, color=text_color, fontweight="bold")

    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels([strategy_labels_mline[s] for s in strategies],
                       fontsize=10, ha="center")
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels(datasets, fontsize=10)
    ax.set_xlabel("Sampling strategy", fontsize=11, labelpad=8)
    ax.set_ylabel("Dataset", fontsize=11)
    ax.set_title(
        "Experiment 1: Mean Macro-F1 by Strategy and Dataset\n"
        "(colour = deviation from per-dataset mean; values = absolute Macro-F1)\n"
        "imbalance-aware companion to the AUC heatmap — surfaces minority-class collapse",
        fontsize=11.5, pad=10,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Deviation from row mean", fontsize=9)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "experiment_1_macro_f1_heatmap.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved: figures/experiment_1_macro_f1_heatmap.png")


def plot_exp2_widening_heatmap(df2):
    """
    Heatmap: 7 metrics × 8 datasets. Cell = spread@10% / spread@100%
    where spread = max - min metric value across 5 strategies (means over seeds).

    Reading:
      ratio > 1  → strategy choice matters MORE at small budget (widening)
      ratio ~ 1  → strategy spread independent of budget
      ratio < 1  → strategy spread NARROWS at small budget (rare)
    """
    metrics = ["macro_auc", "pr_auc", "macro_f1", "balanced_acc",
               "mcc", "log_loss", "accuracy"]
    # All 11 Experiment 2 datasets (7 binary + 4 multiclass).
    # Ordered binaries-first by pool size, then multiclass by class count,
    # mirroring the Chapter 6 retention table.
    datasets = ["higgs", "MiniBooNE", "bank-marketing", "nomao",
                "mozilla4", "adult", "numerai28.6",
                "covertype", "jannis", "connect-4", "volkert"]

    ratios = np.zeros((len(metrics), len(datasets)))
    for i, m in enumerate(metrics):
        for j, ds in enumerate(datasets):
            sub = df2[df2["dataset"] == ds]
            spread_at = {}
            for budget in [0.10, 1.00]:
                bsub = sub[sub["budget_fraction"] == budget]
                means = bsub.groupby("strategy")[m].mean()
                spread_at[budget] = means.max() - means.min() if not means.empty else np.nan
            if spread_at[1.00] and spread_at[1.00] > 0:
                ratios[i, j] = spread_at[0.10] / spread_at[1.00]
            else:
                ratios[i, j] = np.nan

    # Diverging colormap centered at 1.0 (no widening). Cap at 5.0 for colour
    # scale; cells above 5.0 still get their numeric value annotated.
    vmax = 5.0
    # Widened canvas: 8 cols vs old 4, but keep heatmap cells roughly square.
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    im = ax.imshow(ratios, cmap="RdYlGn", vmin=0.5, vmax=vmax,
                   aspect="auto", norm=None)

    for i in range(len(metrics)):
        for j in range(len(datasets)):
            val = ratios[i, j]
            if np.isnan(val):
                ax.text(j, i, "N/A", ha="center", va="center",
                        fontsize=10, color="gray")
                continue
            color = "white" if val >= 4.0 or val < 0.9 else "black"
            ax.text(j, i, f"{val:.1f}x", ha="center", va="center",
                    fontsize=10, color=color, fontweight="bold")

    metric_labels = [METRIC_LABELS[m] for m in metrics]
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(datasets, fontsize=9, rotation=15, ha="right")
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels(metric_labels, fontsize=10)
    ax.set_xlabel("Dataset (left: binary; right: multiclass)",
                  fontsize=11, labelpad=8)
    ax.set_ylabel("Metric", fontsize=11)
    ax.set_title(
        "Strategy Spread Widening at Small Budgets — Experiment 2\n"
        "Cell = spread@10% / spread@100% budget across the 5 strategies; "
        ">1 means strategy choice matters more at small budget\n"
        "(green = widens; red = narrows)",
        fontsize=10.5, pad=10,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Spread ratio (10% / 100% budget)", fontsize=9)
    cbar.ax.axhline(1.0, color="black", linewidth=1.2, linestyle="--")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "classification_exp2_widening.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved: figures/classification_exp2_widening.png")


def plot_exp2_abs_spread_low_budget(df2):
    """
    Companion to plot_exp2_widening_heatmap. Shows the ABSOLUTE
    cross-strategy spread (max - min over the 5 samplers) at the
    low-budget condition (frac = 0.10), per (metric, dataset).

    Reading:
      Cell value is the absolute cost (in metric units) of picking
      the worst sampler instead of the best at the 1,000-row budget.

    Pairs with the widening figure: that one says "spread grew Nx";
    this one says "and in absolute units the spread is now this big."
    No divide-by-near-zero risk — purely the spread at frac=0.10.
    """
    metrics = ["macro_auc", "pr_auc", "macro_f1", "balanced_acc",
               "mcc", "log_loss", "accuracy"]
    datasets = ["higgs", "MiniBooNE", "bank-marketing", "nomao",
                "mozilla4", "adult", "numerai28.6",
                "covertype", "jannis", "connect-4", "volkert"]

    spreads = np.full((len(metrics), len(datasets)), np.nan)
    for i, m in enumerate(metrics):
        for j, ds in enumerate(datasets):
            sub = df2[(df2["dataset"] == ds) &
                      (df2["budget_fraction"] == 0.10)]
            means = sub.groupby("strategy")[m].mean()
            if not means.empty:
                spreads[i, j] = means.max() - means.min()

    # Sequential colour scale; cap to keep small/moderate spreads readable.
    # Practical: 0.05 = 5pp gap (meaningful), 0.10 = 10pp (large),
    # 0.20 = 20pp (catastrophic). Cap at 0.25 — cells above still annotated.
    vmax = 0.25
    # Width scales with dataset count so per-column width stays readable
    fig_w = max(10.5, 1.15 * len(datasets))
    fig, ax = plt.subplots(figsize=(fig_w, 5.5))
    im = ax.imshow(spreads, cmap="YlOrRd", vmin=0.0, vmax=vmax,
                   aspect="auto")

    for i in range(len(metrics)):
        for j in range(len(datasets)):
            val = spreads[i, j]
            if np.isnan(val):
                ax.text(j, i, "N/A", ha="center", va="center",
                        fontsize=10, color="gray")
                continue
            # White text on dark cells, black on light cells.
            color = "white" if val >= 0.15 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=10, color=color, fontweight="bold")

    metric_labels = [METRIC_LABELS[m] for m in metrics]
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(datasets, fontsize=9, rotation=15, ha="right")
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels(metric_labels, fontsize=10)
    ax.set_xlabel("Dataset (left: binary; right: multiclass)",
                  fontsize=11, labelpad=8)
    ax.set_ylabel("Metric", fontsize=11)
    ax.set_title(
        "Absolute Strategy Spread at frac = 0.10 — Experiment 2\n"
        "Cell = max − min metric value across the 5 samplers at "
        "1,000-row budget (mean over four seeds)\n"
        "darker = larger penalty for picking the wrong sampler at low budget",
        fontsize=10.5, pad=10,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Absolute spread at 10% budget (metric units)",
                   fontsize=9)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "classification_exp2_abs_spread_low_budget.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved: figures/classification_exp2_abs_spread_low_budget.png")


def plot_dominance_matrix(wilcox_df):
    """
    5×5 heatmap. Cell (row, col) = number of the 8 metrics on which
    row-strategy SIGNIFICANTLY (p < 0.05) outperforms col-strategy.

    Diagonal is masked out. Upper-right (row better than col) is where
    dominance shows up under the strategy ordering.
    """
    strategies = STRATEGY_ORDER
    n = len(strategies)
    mat = np.full((n, n), np.nan)

    # Build lookup: for each (a, b) pair, count a's significant wins
    lookup = {}
    for _, row in wilcox_df.iterrows():
        a, b = row["strategy_a"], row["strategy_b"]
        if not row["significant_0.05"]:
            continue
        if row["a_wins"] > row["b_wins"]:
            lookup[(a, b)] = lookup.get((a, b), 0) + 1
        elif row["b_wins"] > row["a_wins"]:
            lookup[(b, a)] = lookup.get((b, a), 0) + 1

    for i, ra in enumerate(strategies):
        for j, cb in enumerate(strategies):
            if i == j:
                continue
            mat[i, j] = lookup.get((ra, cb), 0)

    fig, ax = plt.subplots(figsize=(7.5, 6))
    cmap = plt.cm.Greens
    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=8, aspect="auto")

    # Diagonal mask
    for i in range(n):
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1,
                                    facecolor="#dddddd", edgecolor="none"))
        ax.text(i, i, "—", ha="center", va="center",
                fontsize=14, color="#666666")

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            val = int(mat[i, j])
            color = "white" if val >= 6 else "black"
            ax.text(j, i, str(val), ha="center", va="center",
                    fontsize=13, color=color, fontweight="bold")

    labels = [STRATEGY_LABELS_FLAT[s] for s in strategies]
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Loser (column)", fontsize=11, labelpad=8)
    ax.set_ylabel("Winner (row)", fontsize=11)
    ax.set_title(
        "Pairwise Statistical Dominance Across 8 Classification Metrics\n"
        "Cell = # of metrics where row significantly beats column (Wilcoxon p < 0.05)\n"
        "Max possible = 8; cells of 8 = total domination on every metric tested",
        fontsize=11, pad=10,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02,
                        ticks=list(range(0, 9, 2)))
    cbar.set_label("# metrics with p < 0.05 (row > column)", fontsize=9)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "classification_dominance_matrix.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved: figures/classification_dominance_matrix.png")


def plot_ranking_heatmap(pivot):
    """
    Heatmap: 8 metrics (rows) × 5 strategies (cols).
    Color = rank (1 = best per row); annotation = raw metric value.
    Shows visually that all metrics agree on the ranking.
    """
    metrics = [m for m in METRIC_DIRECTION if m in pivot.columns]
    pivot = pivot[metrics]

    # Compute per-metric rank (1 = best)
    rank_data = np.zeros_like(pivot.values, dtype=float)
    for i, metric in enumerate(metrics):
        vals = pivot[metric].values
        ascending = METRIC_DIRECTION[metric] == "lower"
        order = np.argsort(vals) if ascending else np.argsort(-vals)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(vals) + 1)
        rank_data[:, i] = ranks  # rank_data: strategies × metrics

    # Transpose to: metrics × strategies (rows × cols)
    rank_grid = rank_data.T
    val_grid  = pivot.values.T

    n_strategies = len(pivot.index)
    n_metrics = len(metrics)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    im = ax.imshow(rank_grid, cmap="RdYlGn_r", aspect="auto",
                   vmin=1, vmax=n_strategies)

    strategy_labels = [STRATEGY_LABELS_FLAT[s] for s in pivot.index]
    metric_labels   = [METRIC_LABELS[m] for m in metrics]

    for i in range(n_metrics):
        for j in range(n_strategies):
            rank = int(rank_grid[i, j])
            val = val_grid[i, j]
            text_color = "white" if rank == 1 or rank == n_strategies else "black"
            ax.text(j, i, f"{val:.3f}\n(#{rank})",
                    ha="center", va="center",
                    fontsize=9, color=text_color, fontweight="bold")

    ax.set_xticks(range(n_strategies))
    ax.set_xticklabels(strategy_labels, fontsize=10)
    ax.set_yticks(range(n_metrics))
    ax.set_yticklabels(metric_labels, fontsize=10)
    ax.set_xlabel("Sampling strategy", fontsize=11, labelpad=8)
    ax.set_ylabel("Metric", fontsize=11)
    ax.set_title(
        "Overall Strategy Ranking Across 8 Classification Metrics\n"
        "(mean over n = 11 informative datasets, equal weighting; "
        "color = rank within row, 1 = best in green)",
        fontsize=12, pad=10,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02,
                        ticks=list(range(1, n_strategies + 1)))
    cbar.set_label("Rank (1 = best)", fontsize=9)
    cbar.ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "classification_ranking_heatmap.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved: figures/classification_ranking_heatmap.png")


def print_per_class_auc_table(pc_df):
    """Show per-class AUC for covertype and jannis (mean over seeds)."""
    if pc_df.empty:
        return
    print("\n" + "=" * 96)
    print("PER-CLASS AUC — covertype (7 classes) and jannis (4 classes)")
    print("  (mean over 4 seeds; rare classes most diagnostic)")
    print("=" * 96)
    class_cols = [c for c in pc_df.columns if c.startswith("class_")]
    for ds in pc_df["dataset"].unique():
        sub = pc_df[pc_df["dataset"] == ds]
        means = sub.groupby("strategy")[class_cols].mean()
        means = means.loc[[s for s in STRATEGY_ORDER if s in means.index]]
        means.index = [STRATEGY_LABELS_FLAT[s] for s in means.index]
        print(f"\n  {ds}")
        print("  " + means.round(4).to_string().replace("\n", "\n  "))
        means.round(4).to_csv(RESULTS_DIR / f"classification_per_class_auc_{ds}.csv")
        print(f"\n  Saved: classification_per_class_auc_{ds}.csv")


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    print("Loading predictions...")
    with open(RESULTS_DIR / "experiment_1_predictions.pkl", "rb") as f:
        preds1 = pickle.load(f)
    with open(RESULTS_DIR / "experiment_2_predictions.pkl", "rb") as f:
        preds2 = pickle.load(f)
    print(f"Exp1: {len(preds1)} entries  Exp2: {len(preds2)} entries\n")

    test_labels = load_test_labels()

    print("\nComputing Exp1 metrics...")
    df1_full = compute_exp1_metrics(preds1, test_labels)
    print(f"  {len(df1_full)} rows")

    print("Computing Exp2 metrics...")
    df2 = compute_exp2_metrics(preds2, test_labels)
    print(f"  {len(df2)} rows")

    print("Computing per-class AUC for multiclass datasets...")
    pc_df = compute_per_class_auc(preds1, test_labels)
    print(f"  {len(pc_df)} rows")

    # Save raw per-run metrics — includes EXCLUDED datasets for completeness.
    df1_full.to_csv(RESULTS_DIR / "classification_metrics_exp1.csv", index=False)
    df2.to_csv(RESULTS_DIR / "classification_metrics_exp2.csv", index=False)
    pc_df.to_csv(RESULTS_DIR / "classification_per_class_auc_exp1.csv", index=False)
    print("\nSaved raw metrics (all 12 datasets):")
    print("  classification_metrics_exp1.csv")
    print("  classification_metrics_exp2.csv")
    print("  classification_per_class_auc_exp1.csv")

    if EXCLUDED_FROM_MAIN:
        df1 = df1_full[~df1_full["dataset"].isin(EXCLUDED_FROM_MAIN)].reset_index(drop=True)
        n_excluded = df1_full['dataset'].isin(EXCLUDED_FROM_MAIN).sum()
        print(f"\nMain aggregates: {df1['dataset'].nunique()} datasets "
              f"(excluded {EXCLUDED_FROM_MAIN}, {n_excluded} rows dropped).")
    else:
        df1 = df1_full

    # Summary printouts and CSVs
    print_summary(df1)
    write_summary_tables(df1)
    print_metric_highlights(df1)
    ranking = overall_strategy_ranking(df1)
    plot_ranking_heatmap(ranking)
    print_per_class_auc_table(pc_df)

    # Wilcoxon tests on all 8 new metrics
    print("\nRunning pairwise Wilcoxon tests across all metrics...")
    wilcox_df = all_metrics_wilcoxon(df1)
    consensus_df = consensus_summary(wilcox_df)
    wilcox_df.to_csv(RESULTS_DIR / "classification_wilcoxon.csv", index=False)
    consensus_df.to_csv(RESULTS_DIR / "classification_wilcoxon_consensus.csv", index=False)
    print("Saved: classification_wilcoxon.csv, classification_wilcoxon_consensus.csv")
    print_wilcoxon_summary(wilcox_df, consensus_df)
    plot_dominance_matrix(wilcox_df)
    plot_exp2_widening_heatmap(df2)
    plot_exp2_abs_spread_low_budget(df2)
    plot_macro_f1_heatmap(df1)

    print("\n" + "=" * 70)
    print("All classification metrics computed.")
    print("=" * 70)


if __name__ == "__main__":
    run()
