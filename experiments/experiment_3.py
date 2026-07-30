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

EXP3_INNER_STRATEGIES = ["stratified", "random"]
M_VALUES = [1, 3, 5, 10]


def run_experiment_3(
    datasets: dict,
    dry_run: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """
    Fixed budget: diversity vs. size.

    For each dataset × inner_strategy × M × seed:
      - Split train/test (fixed random_state=42, identical to Exp 1)
      - Preprocess features (identical to Exp 1)
      - Cap test set (identical to Exp 1)
      - Draw M independent subsamples of size budget//M
      - Fit TabPFN on each member, average predictions
      - Compute AUC on averaged predictions

    Returns DataFrame with one row per (dataset, inner_strategy, M, seed).
    """
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion

    RESULTS_DIR.mkdir(exist_ok=True)

    records = []
    all_predictions = {}

    dataset_items = list(datasets.items())
    if dry_run:
        dataset_items = dataset_items[:1]

    for ds_name, (X, y) in dataset_items:
        n_classes = len(np.unique(y))
        seeds = SEEDS[:1] if dry_run else SEEDS

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
        total_budget = min(MAX_CONTEXT, pool_size)

        inner_strategies = EXP3_INNER_STRATEGIES
        m_values = M_VALUES[:1] if dry_run else M_VALUES

        for inner_name in inner_strategies:
            sampler = SAMPLERS[inner_name]

            for M in m_values:
                budget_per_member = total_budget // M

                for seed in seeds:
                    print(
                        f"  [{ds_name}] inner={inner_name} M={M} "
                        f"budget_per_member={budget_per_member:,} seed={seed}",
                        end=" ... ",
                        flush=True,
                    )

                    try:
                        member_preds = []
                        total_sampling_time = 0.0
                        total_inference_time = 0.0
                        total_sampled = 0

                        for m in range(M):
                            if M == 1:
                                sub_seed = seed  # match Experiment 1
                            else:
                                sub_seed = seed * 1000 + m

                            # Sample
                            t0 = time.perf_counter()
                            idx = sampler.sample(
                                X_train, y_train, budget_per_member, sub_seed
                            )
                            total_sampling_time += time.perf_counter() - t0

                            X_sample = X_train[idx]
                            y_sample = y_train[idx]
                            total_sampled += len(idx)

                            # Fit TabPFN
                            clf = TabPFNClassifier.create_default_for_version(
                                ModelVersion.V2,
                                n_estimators=TABPFN_N_ESTIMATORS,
                                random_state=sub_seed,
                                device="cpu",
                                ignore_pretraining_limits=True,
                            )
                            t1 = time.perf_counter()
                            clf.fit(X_sample, y_sample)
                            preds = clf.predict_proba(X_test)
                            total_inference_time += time.perf_counter() - t1

                            member_preds.append(preds)

                        # Average predictions
                        ensemble_pred = np.mean(member_preds, axis=0)
                        auc = compute_auc(y_test, ensemble_pred, n_classes)

                        print(
                            f"AUC={auc:.4f}  (members={M}  rows_each={budget_per_member:,}  "
                            f"t_sample={total_sampling_time:.1f}s  "
                            f"t_infer={total_inference_time:.1f}s)"
                        )

                        records.append({
                            "dataset": ds_name,
                            "inner_strategy": inner_name,
                            "M": M,
                            "budget_per_member": budget_per_member,
                            "seed": seed,
                            "auc": auc,
                            "n_sampled_total": total_sampled,
                            "pool_size": pool_size,
                            "sampling_time": round(total_sampling_time, 3),
                            "inference_time": round(total_inference_time, 3),
                            "n_classes": n_classes,
                        })

                        all_predictions[(ds_name, inner_name, M, seed)] = {
                            "ensemble_pred": ensemble_pred,
                            "member_preds": member_preds,
                        }

                    except Exception as exc:
                        print(f"FAILED: {exc}")
                        records.append({
                            "dataset": ds_name,
                            "inner_strategy": inner_name,
                            "M": M,
                            "budget_per_member": budget_per_member,
                            "seed": seed,
                            "auc": np.nan,
                            "n_sampled_total": 0,
                            "pool_size": pool_size,
                            "sampling_time": 0.0,
                            "inference_time": 0.0,
                            "n_classes": n_classes,
                        })

    results_df = pd.DataFrame(records)

    # NEVER auto-write here — the caller (run_experiment_3.py) handles all
    # writing through generalized merge_csv/merge_pkl helpers that correctly
    # scope by --dataset. Auto-writing here would clobber the entire CSV
    # when invoked with --dataset X (same regression we hit in Exp 1/Exp 2).
    return results_df, all_predictions
