#!/usr/bin/env python
"""
Entry point for Experiment 1: Strategy comparison at fixed subsample size.

Usage
-----
# Full run (all 10 datasets × 4 seeds × 5 strategies = 200 TabPFN runs)
python run_experiment_1.py

# Dry run: one dataset (credit-g), one seed — for pipeline validation
python run_experiment_1.py --dry-run

# Single dataset, all seeds
python run_experiment_1.py --dataset bank-marketing

# Single strategy only (merges into existing results without overwriting others)
python run_experiment_1.py --strategy prototype
python run_experiment_1.py --strategy prototype --dataset bank-marketing
"""

import sys
import pickle
import argparse
import pandas as pd
from pathlib import Path

# Make sure imports resolve from the thesis/ root
import os
sys.path.insert(0, os.path.dirname(__file__))

from preprocessing.data_loader import load_all_datasets, load_dataset
from experiments.experiment_1 import run_experiment_1
from configs.config import DATASETS, DEFAULT_TARGET_SIZE
from samplers import SAMPLERS

RESULTS_DIR = Path(__file__).parent / "results"


def print_summary(df: pd.DataFrame) -> None:
    """Print mean AUC ± std per strategy × dataset."""
    if df.empty:
        print("No results to summarise.")
        return

    summary = (
        df.groupby(["dataset", "strategy"])["auc"]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary["auc_str"] = summary.apply(
        lambda r: f"{r['mean']:.4f} ± {r['std']:.4f}", axis=1
    )
    pivot = summary.pivot(index="dataset", columns="strategy", values="auc_str")
    print("\n=== Experiment 1 Summary (mean AUC ± std over seeds) ===")
    print(pivot.to_string())
    print()


def merge_csv(existing_path: Path, new_rows: pd.DataFrame,
              strategy: str = None, dataset: str = None,
              seeds: list = None) -> None:
    """
    Replace rows in existing CSV scoped by `strategy`, `dataset`, and/or `seeds`.

    Behaviour by scope combination (any combination of None/value works):
      - (None, None, None):    full replace — overwrite entire file with new_rows
      - any subset set:        replace only rows matching ALL set scopes
    With `seeds` set, only rows whose `seed` is in the list are replaced;
    rows with other seeds are preserved untouched.
    """
    if not existing_path.exists():
        new_rows.to_csv(existing_path, index=False)
        print(f"  {existing_path.name}: created with {len(new_rows)} rows")
        return

    existing = pd.read_csv(existing_path)

    if strategy is None and dataset is None and seeds is None:
        new_rows.to_csv(existing_path, index=False)
        print(f"  {existing_path.name}: full replace — {len(new_rows)} rows")
        return

    mask = pd.Series([True] * len(existing), index=existing.index)
    if strategy is not None:
        mask &= existing["strategy"] == strategy
    if dataset is not None:
        mask &= existing["dataset"] == dataset
    if seeds is not None:
        mask &= existing["seed"].isin(seeds)
    kept = existing[~mask]
    merged = pd.concat([kept, new_rows], ignore_index=True)
    merged = merged[existing.columns]
    merged.to_csv(existing_path, index=False)

    scope_parts = []
    if dataset is not None: scope_parts.append(f"dataset={dataset}")
    if strategy is not None: scope_parts.append(f"strategy={strategy}")
    if seeds is not None: scope_parts.append(f"seeds={seeds}")
    scope = " ".join(scope_parts) if scope_parts else "all"
    print(f"  {existing_path.name}: {len(kept)} kept + {len(new_rows)} new ({scope})")


def merge_pkl(existing_path: Path, new_preds: dict,
              strategy: str = None, dataset: str = None,
              seeds: list = None) -> None:
    """
    Replace pkl entries scoped by `strategy`, `dataset`, and/or `seeds`. Key
    format is (dataset, strategy, seed) so we filter on positions 0, 1, 2.

    Behaviour mirrors merge_csv.
    """
    if not existing_path.exists():
        with open(existing_path, "wb") as f:
            pickle.dump(new_preds, f)
        print(f"  {existing_path.name}: created with {len(new_preds)} entries")
        return

    with open(existing_path, "rb") as f:
        existing = pickle.load(f)

    if strategy is None and dataset is None and seeds is None:
        with open(existing_path, "wb") as f:
            pickle.dump(new_preds, f)
        print(f"  {existing_path.name}: full replace — {len(new_preds)} entries")
        return

    def matches(k):
        if strategy is not None and k[1] != strategy: return False
        if dataset is not None and k[0] != dataset:   return False
        if seeds is not None and k[2] not in seeds:   return False
        return True

    old_keys = [k for k in existing if matches(k)]
    for k in old_keys:
        del existing[k]
    existing.update(new_preds)
    with open(existing_path, "wb") as f:
        pickle.dump(existing, f)

    scope_parts = []
    if dataset is not None: scope_parts.append(f"dataset={dataset}")
    if strategy is not None: scope_parts.append(f"strategy={strategy}")
    if seeds is not None: scope_parts.append(f"seeds={seeds}")
    scope = " ".join(scope_parts) if scope_parts else "all"
    print(f"  {existing_path.name}: removed {len(old_keys)} old, added {len(new_preds)} new ({scope})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run on credit-g with seed=1 only (pipeline validation)",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=DEFAULT_TARGET_SIZE,
        help=f"Subsample budget (default: {DEFAULT_TARGET_SIZE:,})",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Run a single dataset by name (e.g. 'bank-marketing')",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        choices=list(SAMPLERS.keys()),
        help="Run a single sampling strategy and merge into existing results",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seed list to override config SEEDS "
             "(e.g. '5,6,7,8,9,10'). Merges scoped by seed: existing rows "
             "for other seeds are preserved untouched.",
    )
    args = parser.parse_args()

    seeds_override = (
        [int(s) for s in args.seeds.split(",")] if args.seeds else None
    )

    if args.dry_run:
        print("=== DRY RUN: credit-g, seed=1 ===\n")
        first_name, first_id = next(iter(DATASETS.items()))
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

    strategies = [args.strategy] if args.strategy else None
    if args.strategy:
        print(f"\n=== Running Experiment 1 — strategy: {args.strategy} (target_size={args.target_size:,}) ===\n")
    else:
        print(f"\n=== Running Experiment 1 (target_size={args.target_size:,}) ===\n")

    results_df, predictions = run_experiment_1(
        datasets,
        target_size=args.target_size,
        dry_run=args.dry_run,
        strategies=strategies,
        seeds=seeds_override,
    )

    # Unified write path. The generalized merge_csv/merge_pkl scope the
    # replacement by (strategy, dataset, seeds) per the args. With all None
    # we do a full overwrite (the normal full-run case). With any set we
    # merge — preserving rows outside the scope.
    if not args.dry_run:
        print("\nSaving results...")
        csv_path = RESULTS_DIR / "experiment_1_results.csv"
        pkl_path = RESULTS_DIR / "experiment_1_predictions.pkl"
        merge_csv(csv_path, results_df,
                  strategy=args.strategy, dataset=args.dataset,
                  seeds=seeds_override)
        merge_pkl(pkl_path, predictions,
                  strategy=args.strategy, dataset=args.dataset,
                  seeds=seeds_override)

    print_summary(results_df)


if __name__ == "__main__":
    main()
