"""
Compute the SIGNED calibration gap (mean confidence − accuracy) for every
(strategy, dataset) pair, averaged across seeds.

  Positive value  → model is OVERCONFIDENT
  Negative value  → model is UNDERCONFIDENT
  Magnitude       ≈ ECE (which loses sign)

This answers: "Is Prototype's underconfidence on covertype a one-off,
or a general pattern?"

Outputs:
  - Console table (one row per dataset, columns = strategies)
  - results/figures/calibration_direction_heatmap.png
"""

import sys, os, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

from configs.config import (
    DATASETS, TEST_SIZE, SPLIT_RANDOM_STATE, TEST_MAX_SIZE,
    EXCLUDED_FROM_MAIN,
)
from preprocessing.data_loader import load_dataset

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

STRATEGY_ORDER = ["random", "stratified", "coreset", "prototype", "stratified_coreset"]
STRATEGY_LABELS = {
    "random":             "Random",
    "stratified":         "Stratified",
    "coreset":            "k-Center",
    "prototype":          "Prototype",
    "stratified_coreset": "Per-Class\nk-Center",
}
DATASET_ORDER = [
    "credit-g", "phoneme", "pendigits",
    "mozilla4", "nomao", "bank-marketing", "adult",
    "volkert", "connect-4", "jannis",
    "numerai28.6", "higgs", "MiniBooNE", "covertype",
]
N_BINS = 10


def signed_gap(probs, y, n_bins=N_BINS):
    """
    Returns weighted-average (mean_confidence − accuracy) across bins.
    Positive: overconfident. Negative: underconfident.
    Works for both binary and multiclass using max-probability convention.
    """
    confidence = probs.max(axis=1)
    predicted  = probs.argmax(axis=1)
    correct    = (predicted == y).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(y)
    signed = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidence >= lo) & (confidence < hi) if hi < 1.0 \
               else (confidence >= lo) & (confidence <= hi)
        n = int(mask.sum())
        if n == 0:
            continue
        mean_conf = confidence[mask].mean()
        acc = correct[mask].mean()
        signed += (n / total) * (mean_conf - acc)
    return signed


def load_test_labels_for(name):
    """Reproduce the test labels exactly as experiment_1.py does."""
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
    print("Loading predictions...")
    with open(RESULTS_DIR / "experiment_1_predictions.pkl", "rb") as f:
        preds = pickle.load(f)

    rows = []
    print()
    for ds in DATASET_ORDER:
        if ds in EXCLUDED_FROM_MAIN:
            continue
        try:
            y = load_test_labels_for(ds)
        except Exception as e:
            print(f"  [skip] {ds}: {e}")
            continue
        row = {"dataset": ds}
        for strat in STRATEGY_ORDER:
            gaps = []
            for seed in [1, 2, 3, 4]:
                key = (ds, strat, seed)
                if key not in preds:
                    continue
                probs = preds[key]
                gaps.append(signed_gap(probs, y))
            row[strat] = np.mean(gaps) if gaps else np.nan
        rows.append(row)
        print(f"  {ds:18s}: ", end="")
        for strat in STRATEGY_ORDER:
            v = row[strat]
            sign = "OVER " if v > 0 else "UNDER" if v < 0 else "EXACT"
            print(f"{strat[:8]:>9s}={v:+.3f} ({sign})  ", end="")
        print()

    df = pd.DataFrame(rows).set_index("dataset")
    df = df[STRATEGY_ORDER]

    # Save CSV
    csv_out = RESULTS_DIR / "calibration_direction.csv"
    df.to_csv(csv_out)
    print(f"\nSaved: {csv_out}")

    # ── Heatmap ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 7))
    # Use diverging colormap centered at 0
    vmax = float(np.nanmax(np.abs(df.values)))
    im = ax.imshow(
        df.values, cmap="RdBu_r", aspect="auto",
        vmin=-vmax, vmax=+vmax,
    )
    ax.set_xticks(range(len(STRATEGY_ORDER)))
    ax.set_xticklabels([STRATEGY_LABELS[s] for s in STRATEGY_ORDER], fontsize=11)
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels(df.index, fontsize=10)

    # Annotate each cell with the signed value
    for i in range(len(df.index)):
        for j in range(len(STRATEGY_ORDER)):
            v = df.values[i, j]
            if np.isnan(v):
                continue
            color = "white" if abs(v) > vmax * 0.55 else "black"
            ax.text(j, i, f"{v:+.3f}",
                    ha="center", va="center",
                    color=color, fontsize=10, fontweight="bold")

    ax.set_title(
        "Calibration Direction Heatmap — Signed Gap (mean confidence − accuracy)\n"
        "RED = OVERCONFIDENT     |     BLUE = UNDERCONFIDENT     |     0 = perfectly calibrated",
        fontsize=12, pad=14,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Signed gap  (conf − acc)", fontsize=10)
    fig.tight_layout()
    out = FIGURES_DIR / "calibration_direction_heatmap.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

    # ── Summary verdict ──────────────────────────────────────────────
    print()
    print("=" * 70)
    print("PER-STRATEGY VERDICT (mean signed gap across all datasets)")
    print("=" * 70)
    means = df.mean()
    for strat in STRATEGY_ORDER:
        v = means[strat]
        sign = "OVERCONFIDENT" if v > 0.01 else "UNDERCONFIDENT" if v < -0.01 else "calibrated"
        print(f"  {STRATEGY_LABELS[strat].replace(chr(10),' '):22s}: {v:+.4f}  →  {sign}")

    print()
    print("Per-dataset PROTOTYPE direction (count over/under):")
    proto = df["prototype"]
    over  = (proto > 0.01).sum()
    under = (proto < -0.01).sum()
    cal   = ((proto >= -0.01) & (proto <= 0.01)).sum()
    print(f"  Underconfident: {under} / {len(proto)} datasets")
    print(f"  Overconfident:  {over} / {len(proto)} datasets")
    print(f"  Calibrated:     {cal} / {len(proto)} datasets")


if __name__ == "__main__":
    main()
