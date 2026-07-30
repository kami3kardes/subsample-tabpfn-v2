#!/usr/bin/env python
"""
Entry point for Experiment 3: Fixed Budget — Diversity vs. Size.

Usage
-----
# Full run (2 inner strategies × 14 datasets × 4 seeds, each fitting M∈[1,3,5,10]
#  members: sum(M)=19 TabPFN calls per cell → 19 × 2 × 14 × 4 = 2,128 TabPFN calls)
python run_experiment_3.py

# Dry run: one dataset, M=1, one seed — for pipeline validation
python run_experiment_3.py --dry-run

# Single dataset, all M values and seeds
python run_experiment_3.py --dataset higgs
"""

import sys
import pickle
import argparse
import pandas as pd
from pathlib import Path

import os
sys.path.insert(0, os.path.dirname(__file__))

from preprocessing.data_loader import load_all_datasets, load_dataset
from experiments.experiment_3 import run_experiment_3, M_VALUES, EXP3_INNER_STRATEGIES
from configs.config import DATASETS, MAX_CONTEXT

RESULTS_DIR = Path(__file__).parent / "results"


def merge_csv(existing_path: Path, new_rows: pd.DataFrame,
              dataset: str = None) -> None:
    """
    Replace rows in existing CSV scoped by `dataset`.
      - dataset=None: full replace
      - dataset=Y:    replace all rows where dataset==Y (keep others untouched)
    """
    if not existing_path.exists():
        new_rows.to_csv(existing_path, index=False)
        print(f"  {existing_path.name}: created with {len(new_rows)} rows")
        return
    existing = pd.read_csv(existing_path)
    if dataset is None:
        new_rows.to_csv(existing_path, index=False)
        print(f"  {existing_path.name}: full replace — {len(new_rows)} rows")
        return
    mask = existing["dataset"] == dataset
    kept = existing[~mask]
    merged = pd.concat([kept, new_rows], ignore_index=True)
    merged = merged[existing.columns]
    merged.to_csv(existing_path, index=False)
    print(f"  {existing_path.name}: {len(kept)} kept + {len(new_rows)} new (dataset={dataset})")


def merge_pkl(existing_path: Path, new_preds: dict,
              dataset: str = None) -> None:
    """Pkl analog of merge_csv. Key format: (dataset, inner_strategy, M, seed)."""
    if not existing_path.exists():
        with open(existing_path, "wb") as f:
            pickle.dump(new_preds, f)
        print(f"  {existing_path.name}: created with {len(new_preds)} entries")
        return
    with open(existing_path, "rb") as f:
        existing = pickle.load(f)
    if dataset is None:
        with open(existing_path, "wb") as f:
            pickle.dump(new_preds, f)
        print(f"  {existing_path.name}: full replace — {len(new_preds)} entries")
        return
    old_keys = [k for k in existing if k[0] == dataset]
    for k in old_keys:
        del existing[k]
    existing.update(new_preds)
    with open(existing_path, "wb") as f:
        pickle.dump(existing, f)
    print(f"  {existing_path.name}: removed {len(old_keys)} old, added {len(new_preds)} new (dataset={dataset})")


def print_summary(df: pd.DataFrame) -> None:
    """Print mean AUC ± std per inner_strategy × M × dataset."""
    if df.empty:
        print("No results to summarise.")
        return

    summary = (
        df.groupby(["dataset", "inner_strategy", "M"])["auc"]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary["auc_str"] = summary.apply(
        lambda r: f"{r['mean']:.4f} ± {r['std']:.4f}", axis=1
    )

    print("\n=== Experiment 3 Summary (mean AUC ± std over seeds) ===")
    for inner in sorted(df["inner_strategy"].unique()):
        print(f"\n--- Inner strategy: {inner} ---")
        sub = summary[summary["inner_strategy"] == inner]
        pivot = sub.pivot(index="dataset", columns="M", values="auc_str")
        print(pivot.to_string())
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run one dataset, M=1, one seed (pipeline validation)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Run a single dataset by name (e.g. 'higgs')",
    )
    args = parser.parse_args()

    if args.dry_run:
        first_name, first_id = next(iter(DATASETS.items()))
        print(f"=== DRY RUN: {first_name}, M=1, seed=1 ===\n")
        X, y = load_dataset(first_id, first_name)
        datasets = {first_name: (X, y)}
    elif args.dataset:
        if args.dataset not in DATASETS:
            print(f"Unknown dataset '{args.dataset}'. Available: {list(DATASETS.keys())}")
            sys.exit(1)
        X, y = load_dataset(DATASETS[args.dataset], args.dataset)
        datasets = {args.dataset: (X, y)}
    else:
        print("=== Loading all datasets ===")
        datasets = load_all_datasets()

    total_calls = sum(M for M in M_VALUES) * len(EXP3_INNER_STRATEGIES) * len(datasets) * 4
    print(f"\n=== Running Experiment 3 ({total_calls:,} TabPFN calls) ===\n")
    results_df, predictions = run_experiment_3(
        datasets,
        dry_run=args.dry_run,
    )

    if not args.dry_run:
        print("\nSaving results...")
        csv_path = RESULTS_DIR / "experiment_3_results.csv"
        pkl_path = RESULTS_DIR / "experiment_3_predictions.pkl"
        merge_csv(csv_path, results_df, dataset=args.dataset)
        merge_pkl(pkl_path, predictions, dataset=args.dataset)

    print_summary(results_df)


if __name__ == "__main__":
    main()
