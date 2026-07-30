"""
Calibration Analysis — uses existing predictions, no new TabPFN runs.

Metrics:
  Brier Score: mean((p - y)^2), summed over classes (proper scoring rule)
  ECE: Expected Calibration Error, 10 equal-width bins.
       Binary datasets  (n_classes == 2): binary ECE — bins predict_proba[:,1],
                                          checks fraction-of-positives per bin.
       Multiclass datasets (n_classes > 2): confidence ECE — bins max predicted
                                            probability, checks argmax == true label.
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from configs.config import (
    DATASETS, TEST_SIZE, SPLIT_RANDOM_STATE, TEST_MAX_SIZE, SEEDS,
    DEFAULT_TARGET_SIZE, EXCLUDED_FROM_MAIN,
)
from preprocessing.data_loader import load_dataset
from preprocessing.feature_selector import select_features

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR  = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

STRATEGY_ORDER = ["random", "stratified", "coreset", "prototype", "stratified_coreset"]
STRATEGY_LABELS = {
    "random":             "Random",
    "stratified":         "Stratified",
    "coreset":            "k-Center",
    "prototype":          "Prototype\n(NE)",
    "stratified_coreset": "Per-Class\nk-Center",
}
STRATEGY_LABELS_FLAT = {k: v.replace("\n", " ") for k, v in STRATEGY_LABELS.items()}
STRATEGY_COLORS = {
    "random":             "#1f77b4",
    "stratified":         "#ff7f0e",
    "coreset":            "#2ca02c",
    "prototype":          "#d62728",
    "stratified_coreset": "#9467bd",
}
# Pool-size ordering (same as Exp1 heatmap)
DATASET_ORDER = [
    # Pool ≤ 10K (strategies equivalent)
    "credit-g", "phoneme", "pendigits",
    # Pool > 10K, in increasing size
    "mozilla4", "nomao", "bank-marketing", "adult",
    "volkert", "connect-4", "jannis",
    "numerai28.6", "higgs", "MiniBooNE", "covertype",
]
EXP2_DATASETS  = ["higgs", "MiniBooNE", "covertype", "bank-marketing", "nomao",
                  "connect-4", "jannis", "volkert",
                  "mozilla4", "adult", "numerai28.6"]
N_BINS         = 10


# ── Metric helpers ─────────────────────────────────────────────────────────────

def brier_score(probs, y, n_classes):
    """Multiclass Brier score (sum over classes, mean over samples)."""
    y_onehot = np.zeros_like(probs)
    y_onehot[np.arange(len(y)), y] = 1
    return np.mean(np.sum((probs - y_onehot) ** 2, axis=1))


def ece_binary(probs, y, n_bins=N_BINS):
    """
    Binary ECE using predict_proba[:,1] (positive class probability).
    Bins by predicted positive probability; checks actual fraction of positives.
    Correct for binary classification, especially imbalanced datasets.
    """
    p_pos = probs[:, 1]
    n = len(y)
    ece_val = 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p_pos >= lo) & (p_pos < hi)
        if hi == 1.0:
            mask = (p_pos >= lo) & (p_pos <= hi)
        if mask.sum() == 0:
            continue
        mean_pred = p_pos[mask].mean()
        frac_pos  = y[mask].mean()
        ece_val  += mask.sum() / n * abs(mean_pred - frac_pos)
    return ece_val


def ece_confidence(probs, y, n_bins=N_BINS):
    """
    Confidence-based ECE.
    confidence = max predicted probability; correctness = argmax == true label.
    Appropriate for multiclass datasets (pendigits, volkert, connect-4, jannis, covertype).
    """
    confidence = probs.max(axis=1)
    predicted  = probs.argmax(axis=1)
    correct    = (predicted == y).astype(float)
    n = len(y)
    ece_val = 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidence >= lo) & (confidence < hi)
        if hi == 1.0:
            mask = (confidence >= lo) & (confidence <= hi)
        if mask.sum() == 0:
            continue
        acc  = correct[mask].mean()
        conf = confidence[mask].mean()
        ece_val += mask.sum() / n * abs(conf - acc)
    return ece_val


def compute_ece(probs, y):
    """Dispatch: binary ECE for 2-class, confidence ECE for multiclass."""
    if probs.shape[1] == 2:
        return ece_binary(probs, y)
    else:
        return ece_confidence(probs, y)


def reliability_data_binary(probs, y, n_bins=N_BINS):
    """
    Returns (bin_centres, fraction_pos, counts) for a binary classifier.
    Uses class-1 probability.
    """
    p_pos = probs[:, 1]
    bins  = np.linspace(0.0, 1.0, n_bins + 1)
    centres, frac_pos, counts = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p_pos >= lo) & (p_pos < hi)
        if hi == 1.0:
            mask = (p_pos >= lo) & (p_pos <= hi)
        if mask.sum() == 0:
            centres.append((lo + hi) / 2)
            frac_pos.append(np.nan)
            counts.append(0)
        else:
            centres.append(p_pos[mask].mean())
            frac_pos.append(y[mask].mean())
            counts.append(mask.sum())
    return np.array(centres), np.array(frac_pos), np.array(counts)


# ── Dataset loading (reproduce test sets) ─────────────────────────────────────

def load_test_labels():
    """
    Reproduce the exact y_test used in each experiment by replaying
    the same train/test split + preprocessing as in experiment_1.py.
    Returns dict: {dataset_name: y_test (int array)}
    """
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
            # Remap labels to 0-indexed integers (TabPFN may reorder)
            classes = np.unique(y_test)
            label_map = {c: i for i, c in enumerate(classes)}
            y_test = np.array([label_map[v] for v in y_test])
            test_labels[name] = y_test
            print(f"  {name:20s}: {len(y_test)} test samples  classes={len(classes)}")
        except Exception as exc:
            print(f"  [ERROR] {name}: {exc}")
    return test_labels


# ── Compute per-run metrics ────────────────────────────────────────────────────

def compute_exp1_metrics(preds1, test_labels):
    records = []
    for (ds, strat, seed), probs in preds1.items():
        if ds not in test_labels:
            continue
        y = test_labels[ds]
        n_classes = probs.shape[1]
        bs  = brier_score(probs, y, n_classes)
        ec  = compute_ece(probs, y)
        records.append({
            "dataset": ds, "strategy": strat, "seed": seed,
            "brier": bs, "ece": ec, "n_classes": n_classes,
        })
    return pd.DataFrame(records)


def compute_exp2_metrics(preds2, test_labels):
    records = []
    for (ds, strat, frac, seed), probs in preds2.items():
        if ds not in test_labels:
            continue
        y = test_labels[ds]
        n_classes = probs.shape[1]
        bs  = brier_score(probs, y, n_classes)
        ec  = compute_ece(probs, y)
        records.append({
            "dataset": ds, "strategy": strat,
            "budget_fraction": frac, "seed": seed,
            "brier": bs, "ece": ec,
        })
    return pd.DataFrame(records)


# ── Minority % in each sample (bank-marketing only) ───────────────────────────

def compute_minority_pct(target_ds="bank-marketing"):
    """
    For each (strategy, seed) on target_ds: compute minority class %
    in the 10K context sample. Returns DataFrame.
    """
    from samplers.random_sampler import RandomSampler
    from samplers.stratified_sampler import StratifiedSampler
    from samplers.coreset_sampler import CoresetSampler
    from samplers.prototype_sampler import PrototypeSampler
    from samplers.stratified_coreset import StratifiedCoreset

    samplers = {
        "random":             RandomSampler(),
        "stratified":         StratifiedSampler(),
        "coreset":            CoresetSampler(),
        "prototype":          PrototypeSampler(),
        "stratified_coreset": StratifiedCoreset(),
    }

    did = DATASETS[target_ds]
    X_full, y_full = load_dataset(did, target_ds)
    X_full = select_features(X_full, y_full)

    X_train, _, y_train, _ = train_test_split(
        X_full, y_full, test_size=TEST_SIZE, stratify=y_full,
        random_state=SPLIT_RANDOM_STATE,
    )

    classes, counts = np.unique(y_train, return_counts=True)
    pool_size = len(X_train)
    target_size = min(DEFAULT_TARGET_SIZE, pool_size)
    minority_class = classes[np.argmin(counts)]

    print(f"\nMinority % in {target_ds} sample (pool={pool_size}, target={target_size}):")
    print(f"  Minority class={minority_class}, overall minority%={counts.min()/pool_size*100:.1f}%")
    print(f"  {'Strategy':25s}  {'Seed':>4}  {'Minority%':>10}")

    records = []
    for strat, sampler in samplers.items():
        for seed in SEEDS:
            try:
                X_arr = X_train.values if hasattr(X_train, "values") else X_train
                y_arr = y_train.values if hasattr(y_train, "values") else np.array(y_train)
                idx = sampler.sample(X_arr, y_arr, target_size, seed)
                y_s = y_arr[idx]
                min_pct = np.mean(y_s == minority_class) * 100
                print(f"  {strat:25s}  {seed:>4}  {min_pct:>9.1f}%")
                records.append({"strategy": strat, "seed": seed, "minority_pct": min_pct})
            except Exception as exc:
                print(f"  [ERROR] {strat} seed={seed}: {exc}")
                records.append({"strategy": strat, "seed": seed, "minority_pct": np.nan})

    return pd.DataFrame(records)


# ── Summary tables ─────────────────────────────────────────────────────────────

def print_summary(df1):
    for metric, label in [("brier", "Brier Score"), ("ece", "ECE")]:
        means = df1.groupby(["dataset", "strategy"])[metric].mean()
        pivot = means.unstack("strategy")
        pivot = pivot.loc[
            [d for d in DATASET_ORDER if d in pivot.index],
            [s for s in STRATEGY_ORDER if s in pivot.columns],
        ]
        pivot.columns = [STRATEGY_LABELS_FLAT[s] for s in pivot.columns]
        std_piv = df1.groupby(["dataset", "strategy"])[metric].std().unstack("strategy")
        std_piv = std_piv.loc[
            [d for d in DATASET_ORDER if d in std_piv.index],
            [s for s in STRATEGY_ORDER if s in std_piv.columns],
        ]
        std_piv.columns = [STRATEGY_LABELS_FLAT[s] for s in std_piv.columns]

        print(f"\n{'='*90}")
        print(f"MEAN {label} PER STRATEGY (lower is better)")
        print("="*90)
        print(pivot.round(4).to_string())

        combined = pivot.round(4).astype(str) + " ± " + std_piv.round(4).astype(str)
        combined.to_csv(RESULTS_DIR / f"calibration_{metric}_summary.csv")
        print(f"Saved: calibration_{metric}_summary.csv")


def print_ece_highlights(df1):
    """Print best and worst ECE per dataset (mean over seeds)."""
    means = df1.groupby(["dataset", "strategy"])["ece"].mean().reset_index()
    means["ece_type"] = means["dataset"].apply(
        lambda d: "binary" if df1[df1["dataset"]==d]["n_classes"].iloc[0] == 2 else "confidence"
    )
    print(f"\n{'='*90}")
    print("ECE HIGHLIGHTS — best and worst strategy per dataset")
    print("  (binary ECE for 2-class datasets; "
          "confidence ECE for the 5 multiclass datasets "
          "pendigits, volkert, connect-4, jannis, covertype)")
    print("="*90)
    for ds in DATASET_ORDER:
        sub = means[means["dataset"] == ds].sort_values("ece")
        if sub.empty:
            continue
        ece_type = sub["ece_type"].iloc[0]
        best  = sub.iloc[0]
        worst = sub.iloc[-1]
        print(f"  {ds:20s} [{ece_type:10s}]  "
              f"best: {STRATEGY_LABELS_FLAT[best['strategy']]:20s} ECE={best['ece']:.4f}  |  "
              f"worst: {STRATEGY_LABELS_FLAT[worst['strategy']]:20s} ECE={worst['ece']:.4f}")


# ── Figure A: ECE heatmap ─────────────────────────────────────────────────────

def plot_ece_heatmap(df1):
    means = df1.groupby(["dataset", "strategy"])["ece"].mean().unstack("strategy")
    means = means.loc[
        [d for d in DATASET_ORDER if d in means.index],
        [s for s in STRATEGY_ORDER if s in means.columns],
    ]

    row_means = means.mean(axis=1)
    relative  = means.sub(row_means, axis=0)
    vmax = max(relative.abs().max().max(), 0.002)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(relative.values, cmap="RdYlGn_r",   # reversed: low ECE = good = green
                   vmin=-vmax, vmax=vmax, aspect="auto")

    strategies = list(means.columns)
    datasets   = list(means.index)
    for i, ds in enumerate(datasets):
        for j, strat in enumerate(strategies):
            val = means.loc[ds, strat]
            rel = relative.loc[ds, strat]
            brightness = (rel + vmax) / (2 * vmax)
            text_color = "white" if brightness < 0.25 or brightness > 0.75 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=9, color=text_color, fontweight="bold")

    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels([STRATEGY_LABELS[s] for s in strategies], fontsize=10)
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels(datasets, fontsize=10)
    ax.set_xlabel("Sampling strategy", fontsize=11, labelpad=8)
    ax.set_ylabel("Dataset", fontsize=11)
    ax.set_title(
        "Calibration: Mean ECE by Strategy and Dataset\n"
        "(binary ECE for 2-class datasets; confidence ECE for the 5 multiclass "
        "datasets pendigits, volkert, connect-4, jannis, covertype; lower is better)",
        fontsize=11, pad=10,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Deviation from row mean (red = worse)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "calibration_ece_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved: figures/calibration_ece_heatmap.png")


# ── Figure B: Reliability diagrams for bank-marketing ─────────────────────────

def plot_reliability_diagrams(preds1, test_labels):
    ds   = "bank-marketing"
    seed = 1
    y    = test_labels[ds]

    strategies = [s for s in STRATEGY_ORDER if s in
                  set(k[1] for k in preds1 if k[0] == ds)]

    fig, axes = plt.subplots(1, len(strategies), figsize=(15, 4), sharey=True)

    for ax, strat in zip(axes, strategies):
        key   = (ds, strat, seed)
        probs = preds1[key]
        centres, frac, counts = reliability_data_binary(probs, y)

        valid = ~np.isnan(frac)
        sizes = counts[valid] / counts[valid].sum() * 0.08

        ax.bar(centres[valid], frac[valid], width=sizes + 0.04,
               color=STRATEGY_COLORS[strat], alpha=0.75, label="Calibration")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Perfect")

        for cx, fy in zip(centres[valid], frac[valid]):
            ax.plot([cx, cx], [cx, fy], color="gray", linewidth=1, alpha=0.5)

        # Use binary ECE for bank-marketing (binary dataset)
        ec = ece_binary(probs, y)
        ax.set_title(f"{STRATEGY_LABELS_FLAT[strat]}\nECE={ec:.3f}", fontsize=10)
        ax.set_xlabel("Mean predicted prob. (class 1)", fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel("Fraction of positives", fontsize=10)
    fig.suptitle(
        "Reliability Diagrams — bank-marketing (seed=1, binary ECE)\n"
        "Bars show actual positive rate per predicted-probability bin; diagonal = perfect calibration",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "calibration_reliability_bank_marketing.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved: figures/calibration_reliability_bank_marketing.png")


# ── Figure C: Brier vs AUC scatter ────────────────────────────────────────────

def plot_brier_vs_auc(df1):
    exp1_auc = pd.read_csv(RESULTS_DIR / "experiment_1_results.csv")
    auc_means = exp1_auc.groupby(["dataset", "strategy"])["auc"].mean().reset_index()
    brier_means = df1.groupby(["dataset", "strategy"])["brier"].mean().reset_index()
    merged = auc_means.merge(brier_means, on=["dataset", "strategy"])

    # Below-budget datasets (pool ≤ MAX_CONTEXT) produce bit-identical
    # values across all five strategies because every sampler returns
    # the entire pool. Five coloured dots would stack exactly on top of
    # each other, falsely suggesting a single-strategy outlier. We
    # render these three datasets as a single distinguished marker
    # (hollow grey diamond) with the dataset name annotated.
    BELOW_BUDGET = ["credit-g", "phoneme", "pendigits"]
    informative = merged[~merged["dataset"].isin(BELOW_BUDGET)]
    below_budget_pts = (merged[merged["dataset"].isin(BELOW_BUDGET)]
                        .drop_duplicates(subset=["dataset"])
                        [["dataset", "auc", "brier"]])

    from scipy.spatial import ConvexHull, QhullError
    import numpy as np

    fig, ax = plt.subplots(figsize=(8, 5.5))
    strategies = [s for s in STRATEGY_ORDER if s in informative["strategy"].unique()]

    # 1) Per-dataset clustering: convex hull around each dataset's 5
    #    strategy points + dataset name at the centroid. Tight hulls
    #    (strategies agree) appear as small shapes; loose hulls
    #    (strategies disagree, e.g. covertype) become visually large.
    #    Rendered first / lowest zorder so the coloured dots sit on top.
    for ds in informative["dataset"].unique():
        sub = informative[informative["dataset"] == ds]
        pts = sub[["auc", "brier"]].values
        cx, cy = pts.mean(axis=0)

        if len(pts) >= 3:
            try:
                hull = ConvexHull(pts)
                hull_pts = pts[hull.vertices]
                hull_pts = np.vstack([hull_pts, hull_pts[0]])  # close polygon
                ax.fill(hull_pts[:, 0], hull_pts[:, 1],
                        color="dimgray", alpha=0.07, zorder=1)
                ax.plot(hull_pts[:, 0], hull_pts[:, 1],
                        color="dimgray", alpha=0.45,
                        linewidth=0.8, zorder=2)
            except (QhullError, Exception):
                # Degenerate cases (e.g. all 5 points collinear): fall
                # back to a thin polyline connecting the points.
                pass

        # Dataset label at centroid, slightly offset so it doesn't sit
        # on top of a coloured dot.
        ax.annotate(ds, (cx, cy),
                    xytext=(0, -10), textcoords="offset points",
                    fontsize=7.5, color="black",
                    ha="center", va="top", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.15",
                              fc="white", ec="none", alpha=0.65))

    # 2) Informative datasets: one coloured dot per (dataset, strategy)
    for strat in strategies:
        sub = informative[informative["strategy"] == strat]
        ax.scatter(sub["auc"], sub["brier"],
                   color=STRATEGY_COLORS[strat],
                   label=STRATEGY_LABELS_FLAT[strat],
                   s=55, alpha=0.9, zorder=3, edgecolors="white",
                   linewidths=0.6)

    # 3) Below-budget datasets: single distinguished marker per dataset
    ax.scatter(below_budget_pts["auc"], below_budget_pts["brier"],
               marker="D", s=110, facecolors="none",
               edgecolors="dimgray", linewidths=1.6,
               label="Below-budget dataset\n(all 5 strategies tied)",
               zorder=4)
    for _, row in below_budget_pts.iterrows():
        ax.annotate(row["dataset"], (row["auc"], row["brier"]),
                    xytext=(10, 0), textcoords="offset points",
                    fontsize=7.5, color="dimgray", va="center",
                    fontstyle="italic", zorder=5)

    ax.set_xlabel("Mean AUC-ROC (higher is better)", fontsize=11)
    ax.set_ylabel("Mean Brier Score (lower is better)", fontsize=11)
    ax.set_title(
        "AUC vs. Brier Score by Strategy × Dataset\n"
        "Each grey hull groups the 5 strategy points of one informative "
        "dataset (wide hull ⇒ strategy choice matters);\n"
        "grey diamonds: below-budget datasets where all 5 strategies tie.",
        fontsize=10, pad=10,
    )
    ax.legend(fontsize=8.5, ncol=2, loc="lower left")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "calibration_brier_vs_auc.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved: figures/calibration_brier_vs_auc.png")


# ── Figure D: ECE budget scaling (Experiment 2) ───────────────────────────────

def plot_budget_scaling(df2):
    # Widen proportionally with dataset count so per-panel width stays ~1.75"
    fig_w = max(14, 1.8 * len(EXP2_DATASETS))
    fig, axes = plt.subplots(1, len(EXP2_DATASETS), figsize=(fig_w, 4), sharey=False)

    budget_labels = {0.10: "10%", 0.25: "25%", 0.50: "50%", 1.00: "100%"}
    fracs = sorted(df2["budget_fraction"].unique())

    for ax, ds in zip(axes, EXP2_DATASETS):
        for strat in STRATEGY_ORDER:
            sub = df2[(df2["dataset"] == ds) & (df2["strategy"] == strat)]
            if sub.empty:
                continue
            means = sub.groupby("budget_fraction")["ece"].mean()
            stds  = sub.groupby("budget_fraction")["ece"].std()
            ax.errorbar(
                range(len(fracs)),
                [means.get(f, np.nan) for f in fracs],
                yerr=[stds.get(f, np.nan) for f in fracs],
                marker="o", linewidth=1.8, markersize=5,
                color=STRATEGY_COLORS[strat],
                label=STRATEGY_LABELS_FLAT[strat],
                capsize=3,
            )
        ax.set_title(ds, fontsize=11, fontweight="bold")
        ax.set_xticks(range(len(fracs)))
        ax.set_xticklabels([budget_labels[f] for f in fracs], fontsize=9)
        ax.set_xlabel("Budget fraction", fontsize=9)
        ax.set_ylabel("ECE" if ax == axes[0] else "")
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=9,
               bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(
        "Calibration (ECE) vs. Budget — Experiment 2 Datasets\n"
        "(binary ECE for binary datasets — higgs, MiniBooNE, bank-marketing, nomao;\n"
        "confidence ECE for the 4 multiclass — covertype, jannis, connect-4, volkert)",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "calibration_budget_scaling.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved: figures/calibration_budget_scaling.png")

    means2 = df2.groupby(["dataset", "strategy", "budget_fraction"])["ece"].mean().unstack("budget_fraction")
    means2.to_csv(RESULTS_DIR / "calibration_ece_budget_scaling.csv")
    print("Saved: calibration_ece_budget_scaling.csv")


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
    print(f"  {len(df1_full)} rows computed")

    print("\nComputing Exp2 metrics...")
    df2 = compute_exp2_metrics(preds2, test_labels)
    print(f"  {len(df2)} rows computed")

    # Save raw per-run metrics including EXCLUDED datasets, for completeness.
    df1_full.to_csv(RESULTS_DIR / "calibration_exp1_metrics.csv", index=False)
    df2.to_csv(RESULTS_DIR / "calibration_exp2_metrics.csv", index=False)
    print("Saved: calibration_exp1_metrics.csv (all 13 datasets), "
          "calibration_exp2_metrics.csv")

    if EXCLUDED_FROM_MAIN:
        df1 = df1_full[~df1_full["dataset"].isin(EXCLUDED_FROM_MAIN)].reset_index(drop=True)
        n_excluded = df1_full['dataset'].isin(EXCLUDED_FROM_MAIN).sum()
        print(f"\nMain aggregates: using {df1['dataset'].nunique()} datasets "
              f"(excluded {EXCLUDED_FROM_MAIN}, {n_excluded} rows dropped).")
    else:
        df1 = df1_full

    print_summary(df1)
    print_ece_highlights(df1)

    print("\nGenerating figures...")
    plot_ece_heatmap(df1)
    plot_reliability_diagrams(preds1, test_labels)
    plot_brier_vs_auc(df1)
    plot_budget_scaling(df2)

    # Minority % in bank-marketing samples
    min_pct_df = compute_minority_pct("bank-marketing")
    summary = min_pct_df.groupby("strategy")["minority_pct"].agg(["mean", "std"]).reset_index()
    print(f"\n{'='*60}")
    print("MINORITY % IN bank-marketing SAMPLE (mean over seeds)")
    print("="*60)
    for _, row in summary.iterrows():
        print(f"  {row['strategy']:25s}  {row['mean']:.1f}% ± {row['std']:.1f}%")

    print("\n" + "="*70)
    print("All calibration outputs saved.")
    print("="*70)


if __name__ == "__main__":
    run()
