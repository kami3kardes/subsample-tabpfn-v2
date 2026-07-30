"""
Reliability diagram for covertype (7-class) — confidence-based, like
the existing bank-marketing version but for the multiclass regime.

Each panel shows one strategy. Bars = empirical accuracy per
confidence bin; dashed diagonal = perfect calibration.

  - Bars BELOW the diagonal → model is OVERCONFIDENT
    (says e.g. 90% sure, but only 70% accurate)
  - Bars ABOVE the diagonal → model is UNDERCONFIDENT
    (says e.g. 60% sure, but actually 85% accurate)

Also prints per-bin numbers so we can quantify direction.

Output: results/figures/calibration_reliability_covertype.png
"""

import sys, os, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

from configs.config import (
    DATASETS, TEST_SIZE, SPLIT_RANDOM_STATE, TEST_MAX_SIZE,
)
from preprocessing.data_loader import load_dataset

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
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
N_BINS = 10


def reliability_data_confidence(probs, y, n_bins=N_BINS):
    """
    Multiclass confidence-based reliability.
    confidence = max(probs); correctness = (argmax(probs) == y).
    Returns (centres, accuracy, counts, mean_conf).
    """
    confidence = probs.max(axis=1)
    predicted  = probs.argmax(axis=1)
    correct    = (predicted == y).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centres, accs, counts, mean_confs = [], [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        if hi == 1.0:
            mask = (confidence >= lo) & (confidence <= hi)
        else:
            mask = (confidence >= lo) & (confidence < hi)
        if mask.sum() == 0:
            centres.append((lo + hi) / 2)
            accs.append(np.nan)
            mean_confs.append(np.nan)
            counts.append(0)
        else:
            centres.append((lo + hi) / 2)
            accs.append(correct[mask].mean())
            mean_confs.append(confidence[mask].mean())
            counts.append(int(mask.sum()))
    return (
        np.array(centres),
        np.array(accs),
        np.array(counts),
        np.array(mean_confs),
    )


def ece_from_bins(accs, mean_confs, counts):
    """Compute ECE from binned reliability data."""
    valid = ~np.isnan(accs)
    total = counts[valid].sum()
    if total == 0:
        return 0.0
    return float(
        np.sum(counts[valid] / total * np.abs(mean_confs[valid] - accs[valid]))
    )


def load_covertype_test_labels():
    """Reproduce the test labels exactly as experiment_1.py does."""
    name = "covertype"
    did  = DATASETS[name]
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
    return y_test


def main():
    seed = 1
    print(f"Loading test labels for covertype...")
    y_test = load_covertype_test_labels()
    print(f"  Test set size: {len(y_test)}")
    print(f"  Class distribution: {np.bincount(y_test)}")
    print()

    print(f"Loading predictions...")
    with open(RESULTS_DIR / "experiment_1_predictions.pkl", "rb") as f:
        preds = pickle.load(f)

    print()
    print("=" * 100)
    print(f"COVERTYPE RELIABILITY — SEED {seed}")
    print("=" * 100)

    fig, axes = plt.subplots(1, 5, figsize=(22, 5.2), sharey=True)
    plt.subplots_adjust(wspace=0.18, top=0.78, bottom=0.12, left=0.04, right=0.99)

    for ax, strat in zip(axes, STRATEGY_ORDER):
        key = ("covertype", strat, seed)
        if key not in preds:
            print(f"  [missing] {key}")
            continue
        probs = preds[key]
        centres, accs, counts, mean_confs = reliability_data_confidence(probs, y_test)
        ec = ece_from_bins(accs, mean_confs, counts)

        # Direction diagnostics — over vs under-confident
        valid = ~np.isnan(accs) & (counts > 0)
        # Per-bin signed gap (conf - acc): positive = OVER, negative = UNDER
        signed_gap = mean_confs[valid] - accs[valid]
        weighted_signed_gap = float(
            np.sum(counts[valid] / counts[valid].sum() * signed_gap)
        )
        direction = "OVERCONFIDENT" if weighted_signed_gap > 0 else "UNDERCONFIDENT"

        print(f"\n  {STRATEGY_LABELS[strat]:22s}  ECE = {ec:.4f}  "
              f"|  signed gap (conf − acc): {weighted_signed_gap:+.4f}  →  {direction}")
        print(f"    {'bin':>10s}  {'count':>7s}  {'mean_conf':>10s}  "
              f"{'accuracy':>10s}  {'gap':>10s}  {'direction':>15s}")
        for c, n, mc, a in zip(centres, counts, mean_confs, accs):
            if n == 0:
                continue
            gap = mc - a
            dir_str = "over" if gap > 0 else "under" if gap < 0 else "exact"
            print(f"    [{c-0.05:.2f},{c+0.05:.2f}]  {n:>7d}  "
                  f"{mc:>10.4f}  {a:>10.4f}  {gap:>+10.4f}  {dir_str:>15s}")

        # Visualise — uniform bar widths (one per bin, no overlap)
        valid_mask = ~np.isnan(accs) & (counts > 0)
        BAR_WIDTH = 0.09   # slightly less than bin width 0.1, so bars don't touch

        ax.bar(centres[valid_mask], accs[valid_mask],
               width=BAR_WIDTH, color=STRATEGY_COLORS[strat], alpha=0.78,
               label="Empirical accuracy", edgecolor="black", linewidth=0.5)
        ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Perfect calibration")

        # Show the gap (between confidence and accuracy) as a vertical grey line
        for c, mc, a in zip(centres[valid_mask], mean_confs[valid_mask], accs[valid_mask]):
            ax.plot([c, c], [mc, a], color="gray", linewidth=1.2, alpha=0.7)

        # Annotate sample counts above each bar (so we still encode "where the mass is")
        max_count = counts[valid_mask].max()
        for c, n, a in zip(centres[valid_mask], counts[valid_mask], accs[valid_mask]):
            # Bold the dominant bin so the user can see at a glance where most preds live
            weight = "bold" if n == max_count else "normal"
            ax.text(c, a + 0.04, f"n={n}",
                    fontsize=7, ha="center", va="bottom",
                    color="black", fontweight=weight, alpha=0.85)

        ax.set_title(f"{STRATEGY_LABELS[strat]}\nECE = {ec:.3f}",
                     fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Mean predicted confidence", fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel("Empirical accuracy", fontsize=11)
    axes[-1].legend(fontsize=9, loc="lower right", framealpha=0.9)

    fig.suptitle(
        f"Reliability Diagrams — covertype  (seed = {seed},  7-class,  confidence-based ECE)\n"
        f"Bars BELOW dashed diagonal = OVERCONFIDENT     "
        f"|     Bars ABOVE dashed diagonal = UNDERCONFIDENT",
        fontsize=13, y=0.99,
    )
    out = FIGURES_DIR / "calibration_reliability_covertype.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
