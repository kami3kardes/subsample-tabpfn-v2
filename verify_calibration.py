"""
Calibration verification script.

Checks:
1. Calibration CSV files are present and contain expected ECE values
2. Figure files have modification timestamps after the ECE fix
3. Correct ECE method is used per dataset (binary vs confidence)
4. Spot-check: recompute ECE from saved predictions and compare to CSV
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from configs.config import DATASETS, TEST_SIZE, SPLIT_RANDOM_STATE, TEST_MAX_SIZE
from preprocessing.data_loader import load_dataset

RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# Commit time of the ECE fix (6e11df5, 2026-02-22)
FIX_COMMIT_TIME = datetime(2026, 2, 22, tzinfo=timezone.utc)

PASS = "  [PASS]"
FAIL = "  [FAIL]"


# ── 1. Check CSV files ─────────────────────────────────────────────────────────

def check_csvs():
    print("=" * 70)
    print("1. CALIBRATION CSV FILES")
    print("=" * 70)

    # ECE summary
    ece_path = RESULTS_DIR / "calibration_ece_summary.csv"
    if not ece_path.exists():
        print(f"{FAIL} {ece_path.name} not found")
        return

    ece_df = pd.read_csv(ece_path, index_col=0)
    print(f"{PASS} Found {ece_path.name}")
    print(f"\n  ECE summary (bank-marketing row):")
    if "bank-marketing" in ece_df.index:
        row = ece_df.loc["bank-marketing"]
        for col, val in row.items():
            # values are "mean ± std" strings
            mean_val = float(str(val).split(" ")[0])
            flag = ""
            if "Prototype" in col:
                flag = " ← should be ~0.027, NOT ~0.007"
                status = PASS if mean_val > 0.020 else FAIL
            else:
                status = ""
            print(f"    {col:22s}: {val}{flag} {status}")
    else:
        print(f"{FAIL} bank-marketing not in index")

    # Brier summary
    brier_path = RESULTS_DIR / "calibration_brier_summary.csv"
    if brier_path.exists():
        print(f"\n{PASS} Found {brier_path.name}")
    else:
        print(f"\n{FAIL} {brier_path.name} not found")

    # Exp1 metrics raw
    m1_path = RESULTS_DIR / "calibration_exp1_metrics.csv"
    if m1_path.exists():
        m1 = pd.read_csv(m1_path)
        n = len(m1)
        expected = 200  # 10 datasets × 5 strategies × 4 seeds
        status = PASS if n == expected else FAIL
        print(f"\n{status} {m1_path.name}: {n} rows (expected {expected})")

        # Show n_classes per dataset to confirm binary vs multiclass mapping
        nc = m1.groupby("dataset")["n_classes"].first().sort_values()
        print(f"\n  n_classes per dataset:")
        for ds, nc_val in nc.items():
            ece_type = "binary" if nc_val == 2 else "confidence"
            print(f"    {ds:20s}: {nc_val} classes  → {ece_type} ECE")
    else:
        print(f"\n{FAIL} {m1_path.name} not found")


# ── 2. Figure timestamps ───────────────────────────────────────────────────────

def check_figure_timestamps():
    print("\n" + "=" * 70)
    print("2. FIGURE MODIFICATION TIMESTAMPS")
    print("=" * 70)

    figures = [
        "calibration_ece_heatmap.png",
        "calibration_brier_vs_auc.png",
        "calibration_reliability_bank_marketing.png",
        "calibration_budget_scaling.png",
    ]

    print(f"  (ECE fix commit time: {FIX_COMMIT_TIME.strftime('%Y-%m-%d %H:%M UTC')})\n")

    for fname in figures:
        fpath = FIGURES_DIR / fname
        if not fpath.exists():
            print(f"{FAIL} {fname}: NOT FOUND")
            continue
        mtime = datetime.fromtimestamp(fpath.stat().st_mtime, tz=timezone.utc)
        after_fix = mtime > FIX_COMMIT_TIME
        status = PASS if after_fix else FAIL
        print(f"{status} {fname}")
        print(f"         last modified: {mtime.strftime('%Y-%m-%d %H:%M UTC')}")


# ── 3. ECE method per dataset ─────────────────────────────────────────────────

def check_ece_methods():
    print("\n" + "=" * 70)
    print("3. ECE COMPUTATION METHOD PER DATASET")
    print("=" * 70)

    m1_path = RESULTS_DIR / "calibration_exp1_metrics.csv"
    if not m1_path.exists():
        print(f"{FAIL} calibration_exp1_metrics.csv not found")
        return

    m1 = pd.read_csv(m1_path)
    nc_map = m1.groupby("dataset")["n_classes"].first().to_dict()

    expected_binary = {
        "credit-g", "phoneme", "mozilla4", "bank-marketing",
        "adult", "higgs", "MiniBooNE", "numerai28.6",
    }
    expected_confidence = {"covertype", "jannis"}

    all_ok = True
    print(f"  {'Dataset':20s}  {'n_classes':>9}  {'Method':12s}  Status")
    print(f"  {'-'*20}  {'-'*9}  {'-'*12}  ------")
    for ds in sorted(nc_map):
        nc = nc_map[ds]
        method = "binary" if nc == 2 else "confidence"
        if nc == 2:
            ok = ds in expected_binary
        else:
            ok = ds in expected_confidence
        status = PASS if ok else FAIL
        if not ok:
            all_ok = False
        print(f"  {ds:20s}  {nc:>9}  {method:12s}  {status}")

    print()
    if all_ok:
        print(f"{PASS} All datasets use the correct ECE method")
    else:
        print(f"{FAIL} Some datasets use the wrong ECE method — check code")


# ── 4. Spot-check: recompute ECE for bank-marketing prototype seed=1 ──────────

def _ece_binary(probs, y, n_bins=10):
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
        ece_val += mask.sum() / n * abs(p_pos[mask].mean() - y[mask].mean())
    return ece_val


def spot_check_ece():
    print("\n" + "=" * 70)
    print("4. SPOT-CHECK: bank-marketing prototype seed=1")
    print("=" * 70)

    # Load saved predictions
    preds_path = RESULTS_DIR / "experiment_1_predictions.pkl"
    if not preds_path.exists():
        print(f"{FAIL} experiment_1_predictions.pkl not found")
        return

    with open(preds_path, "rb") as f:
        preds = pickle.load(f)

    key = ("bank-marketing", "prototype", 1)
    if key not in preds:
        print(f"{FAIL} Key {key} not in predictions")
        return

    probs = preds[key]
    print(f"  Predictions shape: {probs.shape}  (n_test × n_classes)")
    assert probs.shape[1] == 2, "Expected 2 classes for bank-marketing"

    # Reconstruct y_test
    did = DATASETS["bank-marketing"]
    X, y = load_dataset(did, "bank-marketing")
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SPLIT_RANDOM_STATE
    )
    if len(X_test) > TEST_MAX_SIZE:
        sss = StratifiedShuffleSplit(
            n_splits=1, train_size=TEST_MAX_SIZE, random_state=SPLIT_RANDOM_STATE
        )
        idx, _ = next(sss.split(X_test, y_test))
        y_test = y_test[idx]
    classes = np.unique(y_test)
    label_map = {c: i for i, c in enumerate(classes)}
    y_test = np.array([label_map[v] for v in y_test])

    # Recompute binary ECE
    recomputed_ece = _ece_binary(probs, y_test)

    # Load from CSV
    m1 = pd.read_csv(RESULTS_DIR / "calibration_exp1_metrics.csv")
    csv_ece = m1[
        (m1["dataset"] == "bank-marketing") &
        (m1["strategy"] == "prototype") &
        (m1["seed"] == 1)
    ]["ece"].values

    if len(csv_ece) == 0:
        print(f"{FAIL} No matching row found in CSV")
        return

    csv_val = csv_ece[0]
    delta = abs(recomputed_ece - csv_val)
    match = delta < 1e-9

    print(f"\n  Recomputed binary ECE : {recomputed_ece:.6f}")
    print(f"  CSV value             : {csv_val:.6f}")
    print(f"  Difference            : {delta:.2e}")

    if match:
        print(f"\n{PASS} Recomputed ECE matches CSV exactly")
    else:
        print(f"\n{FAIL} Mismatch! (delta={delta:.2e})")

    # Sanity check: old confidence ECE would have been ~0.007
    confidence_ece = 0.0
    n = len(y_test)
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y_test).astype(float)
    bins = np.linspace(0.0, 1.0, 11)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf >= lo) & (conf < hi)
        if hi == 1.0:
            mask = (conf >= lo) & (conf <= hi)
        if mask.sum():
            confidence_ece += mask.sum() / n * abs(conf[mask].mean() - correct[mask].mean())

    print(f"\n  For comparison — OLD confidence ECE : {confidence_ece:.6f}")
    if recomputed_ece > confidence_ece + 0.005:
        print(f"{PASS} Binary ECE ({recomputed_ece:.4f}) correctly higher than "
              f"confidence ECE ({confidence_ece:.4f}) — imbalance artifact removed")
    else:
        print(f"{FAIL} Binary ECE unexpectedly not higher than confidence ECE")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("CALIBRATION VERIFICATION")
    print("=" * 70)
    check_csvs()
    check_figure_timestamps()
    check_ece_methods()
    spot_check_ece()
    print("\n" + "=" * 70)
    print("Done.")
    print("=" * 70 + "\n")
