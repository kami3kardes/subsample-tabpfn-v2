#!/usr/bin/env python
"""
Entry point for Experiment 2: Subsample Size Scaling Curves.

Usage
-----
# Full run (4 datasets × 4 budgets × 4 seeds × 5 strategies = 320 TabPFN runs)
python run_experiment_2.py

# Dry run: one dataset, one budget, one seed — for pipeline validation
python run_experiment_2.py --dry-run

# Single dataset, all budgets and seeds
python run_experiment_2.py --dataset higgs

# Single strategy only (merges into existing results without overwriting others)
python run_experiment_2.py --strategy prototype
python run_experiment_2.py --strategy prototype --dataset higgs
"""

import sys
import pickle
import argparse
import pandas as pd
from pathlib import Path

import os
sys.path.insert(0, os.path.dirname(__file__))

from preprocessing.data_loader import load_dataset
from experiments.experiment_2 import run_experiment_2, EXP2_DATASETS, BUDGET_FRACTIONS
from configs.config import DATASETS, MAX_CONTEXT
from samplers import SAMPLERS

RESULTS_DIR = Path(__file__).parent / "results"


def print_summary(df: pd.DataFrame) -> None:
    """Print mean AUC ± std per strategy × budget × dataset."""
    if df.empty:
        print("No results to summarise.")
        return

    summary = (
        df.groupby(["dataset", "budget_fraction", "strategy"])["auc"]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary["auc_str"] = summary.apply(
        lambda r: f"{r['mean']:.4f} ± {r['std']:.4f}", axis=1
    )

    print("\n=== Experiment 2 Summary (mean AUC ± std over seeds) ===")
    for ds in sorted(df["dataset"].unique()):
        ds_data = summary[summary["dataset"] == ds]
        pivot = ds_data.pivot(
            index="budget_fraction", columns="strategy", values="auc_str"
        )
        print(f"\n--- {ds} ---")
        print(pivot.to_string())
    print()


def merge_csv(existing_path: Path, new_rows: pd.DataFrame,
              strategy: str = None, dataset: str = None) -> None:
    """
    Replace rows in existing CSV scoped by `strategy` and/or `dataset`.
    Both None: full overwrite. Either set: scoped replace.
    Note: pkl keys are (dataset, strategy, budget_fraction, seed); we filter
    on positions 0 (dataset) and 1 (strategy).
    """
    if not existing_path.exists():
        new_rows.to_csv(existing_path, index=False)
        print(f"  {existing_path.name}: created with {len(new_rows)} rows")
        return
    existing = pd.read_csv(existing_path)
    if strategy is None and dataset is None:
        new_rows.to_csv(existing_path, index=False)
        print(f"  {existing_path.name}: full replace — {len(new_rows)} rows")
        return
    mask = pd.Series([True] * len(existing), index=existing.index)
    if strategy is not None:
        mask &= existing["strategy"] == strategy
    if dataset is not None:
        mask &= existing["dataset"] == dataset
    kept = existing[~mask]
    merged = pd.concat([kept, new_rows], ignore_index=True)
    merged = merged[existing.columns]
    merged.to_csv(existing_path, index=False)
    scope_parts = []
    if dataset is not None: scope_parts.append(f"dataset={dataset}")
    if strategy is not None: scope_parts.append(f"strategy={strategy}")
    scope = " ".join(scope_parts) if scope_parts else "all"
    print(f"  {existing_path.name}: {len(kept)} kept + {len(new_rows)} new ({scope})")


def merge_pkl(existing_path: Path, new_preds: dict,
              strategy: str = None, dataset: str = None) -> None:
    """Pkl analog of merge_csv. Key format: (dataset, strategy, budget, seed)."""
    if not existing_path.exists():
        with open(existing_path, "wb") as f:
            pickle.dump(new_preds, f)
        print(f"  {existing_path.name}: created with {len(new_preds)} entries")
        return
    with open(existing_path, "rb") as f:
        existing = pickle.load(f)
    if strategy is None and dataset is None:
        with open(existing_path, "wb") as f:
            pickle.dump(new_preds, f)
        print(f"  {existing_path.name}: full replace — {len(new_preds)} entries")
        return
    def matches(k):
        if strategy is not None and k[1] != strategy: return False
        if dataset is not None and k[0] != dataset:   return False
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
    scope = " ".join(scope_parts) if scope_parts else "all"
    print(f"  {existing_path.name}: removed {len(old_keys)} old, added {len(new_preds)} new ({scope})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run one dataset, one budget, one seed (pipeline validation)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Run a single dataset by name (e.g. 'higgs')",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        choices=list(SAMPLERS.keys()),
        help="Run a single sampling strategy and merge into existing results",
    )
    args = parser.parse_args()

    if args.dry_run:
        first_name = EXP2_DATASETS[0]
        print(f"=== DRY RUN: {first_name}, budget=10%, seed=1 ===\n")
        X, y = load_dataset(DATASETS[first_name], first_name)
        datasets = {first_name: (X, y)}
    elif args.dataset:
        if args.dataset not in DATASETS:
            print(f"Unknown dataset '{args.dataset}'. Available: {list(DATASETS.keys())}")
            sys.exit(1)
        if args.dataset not in EXP2_DATASETS:
            print(f"Warning: '{args.dataset}' is not in EXP2_DATASETS {EXP2_DATASETS}")
        X, y = load_dataset(DATASETS[args.dataset], args.dataset)
        datasets = {args.dataset: (X, y)}
    else:
        print("=== Loading Experiment 2 datasets ===")
        datasets = {}
        for name in EXP2_DATASETS:
            X, y = load_dataset(DATASETS[name], name)
            datasets[name] = (X, y)

    strategies = [args.strategy] if args.strategy else None
    budgets_str = str([int(MAX_CONTEXT * f) for f in BUDGET_FRACTIONS])
    if args.strategy:
        print(f"\n=== Running Experiment 2 — strategy: {args.strategy} (budgets: {budgets_str}) ===\n")
    else:
        print(f"\n=== Running Experiment 2 (budgets: {budgets_str}) ===\n")

    results_df, predictions = run_experiment_2(
        datasets,
        dry_run=args.dry_run,
        strategies=strategies,
    )

    if not args.dry_run:
        print("\nSaving results...")
        csv_path = RESULTS_DIR / "experiment_2_results.csv"
        pkl_path = RESULTS_DIR / "experiment_2_predictions.pkl"
        merge_csv(csv_path, results_df,
                  strategy=args.strategy, dataset=args.dataset)
        merge_pkl(pkl_path, predictions,
                  strategy=args.strategy, dataset=args.dataset)

    print_summary(results_df)


if __name__ == "__main__":
    main()
