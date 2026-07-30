import time
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

from configs.config import (
    SEEDS,
    TEST_SIZE,
    SPLIT_RANDOM_STATE,
    MAX_CONTEXT,
    TABPFN_N_ESTIMATORS,
    TEST_MAX_SIZE,
)
from preprocessing.feature_selector import select_features
from samplers import SAMPLERS
from analysis.metrics import compute_auc

RESULTS_DIR = Path(__file__).parent.parent / "results"

EXP2_DATASETS = ["higgs", "MiniBooNE", "covertype", "bank-marketing", "nomao",
                 "connect-4", "jannis", "volkert",
                 "mozilla4", "adult", "numerai28.6"]
BUDGET_FRACTIONS = [0.10, 0.25, 0.50, 1.00]


def run_experiment_2(
    datasets: dict,
    dry_run: bool = False,
    strategies: list = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Subsample size scaling curves.

    For each dataset × budget_fraction × seed × strategy:
      - Split train/test (fixed random_state=42, identical to Exp 1)
      - Preprocess features (identical to Exp 1)
      - Cap test set (identical to Exp 1)
      - Sample `budget` rows from train pool
      - Fit TabPFN v2, predict, compute AUC

    Args:
        strategies: list of strategy names to run (default None = all).
                    When a subset is given the CSV/pkl are NOT overwritten;
                    the caller is responsible for merging.

    Returns (DataFrame, predictions_dict) with one row per (dataset, strategy, budget_fraction, seed).
    """
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion

    RESULTS_DIR.mkdir(exist_ok=True)

    records = []
    all_predictions = {}

    samplers_to_run = {k: v for k, v in SAMPLERS.items()
                       if strategies is None or k in strategies}

    dataset_items = list(datasets.items())

    for ds_name, (X, y) in dataset_items:
        n_classes = len(np.unique(y))
        seeds = SEEDS[:1] if dry_run else SEEDS
        budgets = BUDGET_FRACTIONS[:1] if dry_run else BUDGET_FRACTIONS

        # Fixed train/test split (identical to Exp 1)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            stratify=y,
            random_state=SPLIT_RANDOM_STATE,
        )

        # Cap test set (identical to Exp 1)
        if len(X_test) > TEST_MAX_SIZE:
            sss = StratifiedShuffleSplit(n_splits=1, train_size=TEST_MAX_SIZE,
                                         random_state=SPLIT_RANDOM_STATE)
            test_idx, _ = next(sss.split(X_test, y_test))
            X_test = X_test[test_idx]
            y_test = y_test[test_idx]

        # Feature preprocessing (identical to Exp 1)
        X_train = select_features(X_train, y_train)
        X_test = X_test[:, :X_train.shape[1]]

        pool_size = len(X_train)

        for frac in budgets:
            budget = int(MAX_CONTEXT * frac)
            print(f"\n--- {ds_name} | budget={budget:,} ({frac:.0%}) ---")

            for seed in seeds:
                for strategy_name, sampler in samplers_to_run.items():
                    print(
                        f"  [{ds_name}] frac={frac} seed={seed} "
                        f"strategy={strategy_name} budget={budget:,}",
                        end=" ... ",
                        flush=True,
                    )

                    try:
                        # Sampling
                        t0 = time.perf_counter()
                        idx = sampler.sample(X_train, y_train, budget, seed)
                        sampling_time = time.perf_counter() - t0

                        X_sample = X_train[idx]
                        y_sample = y_train[idx]
                        n_sampled = len(idx)

                        # Fit TabPFN v2
                        clf = TabPFNClassifier.create_default_for_version(
                            ModelVersion.V2,
                            n_estimators=TABPFN_N_ESTIMATORS,
                            random_state=seed,
                            device="cpu",
                            ignore_pretraining_limits=True,
                        )
                        t1 = time.perf_counter()
                        clf.fit(X_sample, y_sample)
                        proba = clf.predict_proba(X_test)
                        inference_time = time.perf_counter() - t1

                        # AUC
                        auc = compute_auc(y_test, proba, n_classes)

                        print(
                            f"AUC={auc:.4f}  (sampled={n_sampled:,}  "
                            f"t_sample={sampling_time:.1f}s  t_infer={inference_time:.1f}s)"
                        )

                        records.append({
                            "dataset": ds_name,
                            "strategy": strategy_name,
                            "budget_fraction": frac,
                            "budget": budget,
                            "seed": seed,
                            "auc": auc,
                            "n_sampled": n_sampled,
                            "pool_size": pool_size,
                            "sampling_time": round(sampling_time, 3),
                            "inference_time": round(inference_time, 3),
                            "n_classes": n_classes,
                        })

                        all_predictions[(ds_name, strategy_name, frac, seed)] = proba

                    except Exception as exc:
                        print(f"FAILED: {exc}")
                        records.append({
                            "dataset": ds_name,
                            "strategy": strategy_name,
                            "budget_fraction": frac,
                            "budget": budget,
                            "seed": seed,
                            "auc": np.nan,
                            "n_sampled": 0,
                            "pool_size": pool_size,
                            "sampling_time": 0.0,
                            "inference_time": 0.0,
                            "n_classes": n_classes,
                        })

    results_df = pd.DataFrame(records)

    # NEVER auto-write — caller handles all writes via merge helpers
    # (mirrors the fix applied to experiment_1.py to prevent the regression
    # where `--dataset X` without --strategy overwrote the entire CSV).
    return results_df, all_predictions
