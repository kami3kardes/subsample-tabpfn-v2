"""
Sampling strategy visualisation on bank-marketing (PCA projection).

Outputs
-------
  results/figures/bank_marketing_sampling_1k.png
      Figure A — all 5 strategies at 1K budget (6-panel single row)
  results/figures/bank_marketing_budget_scaling.png
      Figure B — stratified vs stratified k-Center at 4 budgets (2×4 grid)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from preprocessing.data_loader import load_dataset
from preprocessing.feature_selector import select_features
from samplers import SAMPLERS
from configs.config import DATASETS, SPLIT_RANDOM_STATE, TEST_SIZE

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

MAJ_COLOR = "#2196F3"   # blue
MIN_COLOR = "#FF5722"   # red-orange


def _load_exp2_aucs(results_dir: Path, dataset: str = "bank-marketing", seed: int = 1):
    """
    Load AUC values for visualisation annotations directly from the
    Experiment 2 results CSV so they stay in sync after reruns.

    Returns:
        aucs_1k     : {strategy: auc}  at 10% budget (1K rows), given seed
        aucs_scaling: {strategy: {budget_rows: auc}}  all budgets, given seed
    """
    csv_path = results_dir / "experiment_2_results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Experiment 2 results not found at {csv_path}. "
            "Run run_experiment_2.py first."
        )
    df = pd.read_csv(csv_path)
    df = df[(df["dataset"] == dataset) & (df["seed"] == seed)]

    aucs_1k = (
        df[df["budget_fraction"] == 0.10]
        .set_index("strategy")["auc"]
        .to_dict()
    )

    scaling_strategies = ["stratified", "stratified_coreset"]
    aucs_scaling = {}
    for strat in scaling_strategies:
        sub = df[df["strategy"] == strat].set_index("budget")["auc"].to_dict()
        # keys are int budget sizes
        aucs_scaling[strat] = {int(k): v for k, v in sub.items()}

    return aucs_1k, aucs_scaling

STRATEGY_LABELS = {
    "random":            "Random",
    "stratified":        "Stratified",
    "coreset":           "k-Center",
    "prototype":         "Prototype (NE)",
    "stratified_coreset": "Per-Class k-Center",
}

BUDGETS = [1000, 2500, 5000, 10000]


# ── Data loading ─────────────────────────────────────────────────────────────

def load_bank_marketing():
    """Load, preprocess, split, and feature-select bank-marketing training pool."""
    X, y = load_dataset(DATASETS["bank-marketing"], "bank-marketing")
    X_train, _, y_train, _ = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=SPLIT_RANDOM_STATE,
    )
    X_train = select_features(X_train, y_train)
    return X_train, y_train


# ── PCA + axis helpers ────────────────────────────────────────────────────────

def fit_pca2(X_pool):
    pca = PCA(n_components=2, random_state=0)
    Z = pca.fit_transform(X_pool.astype(np.float32))
    return Z


def shared_limits(Z, pad=0.05):
    """1st–99th percentile limits with proportional padding."""
    x1, x99 = np.percentile(Z[:, 0], [1, 99])
    y1, y99 = np.percentile(Z[:, 1], [1, 99])
    xpad = (x99 - x1) * pad
    ypad = (y99 - y1) * pad
    return (x1 - xpad, x99 + xpad), (y1 - ypad, y99 + ypad)


def detect_majority(y):
    """Return (maj_label, min_label) from class counts."""
    counts = np.bincount(y)
    maj = int(np.argmax(counts))
    labels = list(range(len(counts)))
    min_ = [l for l in labels if l != maj][0]
    return maj, min_


def class_colors(y, maj_label):
    return np.where(y == maj_label, MAJ_COLOR, MIN_COLOR)


def subtitle_text(y_sel, maj_label, min_label, auc=None):
    """One-line summary: (n_maj / n_min, X.X% min)  AUC=X.XXX"""
    counts = np.bincount(y_sel, minlength=max(maj_label, min_label) + 1)
    n_maj = counts[maj_label]
    n_min = counts[min_label]
    pct_min = 100.0 * n_min / len(y_sel)
    base = f"({n_maj:,} / {n_min:,},  {pct_min:.1f}% min)"
    if auc is not None:
        return f"{base}   AUC = {auc:.3f}"
    return base


# ── Generic panel drawing ─────────────────────────────────────────────────────

def draw_panel(ax, Z_bg, y_bg, Z_sel, y_sel, maj_label, min_label,
               title, sub, xlim, ylim, s_sel=12, bg_alpha=0.03):
    # Background: full pool, faded
    ax.scatter(
        Z_bg[:, 0], Z_bg[:, 1],
        c=class_colors(y_bg, maj_label),
        s=3, alpha=bg_alpha, linewidths=0, rasterized=True,
    )
    # Selected points
    ax.scatter(
        Z_sel[:, 0], Z_sel[:, 1],
        c=class_colors(y_sel, maj_label),
        s=s_sel, alpha=0.75, linewidths=0, rasterized=True,
    )
    ax.set_title(title, fontsize=11, fontweight="bold", pad=3)
    ax.set_xlabel(sub, fontsize=8, labelpad=3)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)


def pool_panel(ax, Z_bg, y_bg, maj_label, min_label, xlim, ylim):
    """Special panel for the full pool (no selection highlight)."""
    counts = np.bincount(y_bg)
    n_maj = counts[maj_label]
    n_min = counts[min_label]
    pct_min = 100.0 * n_min / len(y_bg)
    ax.scatter(
        Z_bg[:, 0], Z_bg[:, 1],
        c=class_colors(y_bg, maj_label),
        s=3, alpha=0.06, linewidths=0, rasterized=True,
    )
    ax.set_title("Full Pool", fontsize=11, fontweight="bold", pad=3)
    ax.set_xlabel(
        f"({n_maj:,} / {n_min:,},  {pct_min:.1f}% min)   n = {len(y_bg):,}",
        fontsize=8, labelpad=3,
    )
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)


def legend_patches():
    return [
        Patch(facecolor=MAJ_COLOR, label="Majority"),
        Patch(facecolor=MIN_COLOR, label="Minority"),
    ]


# ── Figure A ──────────────────────────────────────────────────────────────────

def figure_a(Z_pool, y_pool, samples_1k, maj_label, min_label, xlim, ylim, aucs_1k):
    """All 5 strategies at 1K budget — single row of 6 panels."""
    fig, axes = plt.subplots(1, 6, figsize=(26, 4.2))

    # Panel 0: full pool
    pool_panel(axes[0], Z_pool, y_pool, maj_label, min_label, xlim, ylim)

    # Panels 1-5: each strategy
    for ax, (name, idx) in zip(axes[1:], samples_1k.items()):
        Z_sel = Z_pool[idx]
        y_sel = y_pool[idx]
        draw_panel(
            ax, Z_pool, y_pool, Z_sel, y_sel, maj_label, min_label,
            title=STRATEGY_LABELS[name],
            sub=subtitle_text(y_sel, maj_label, min_label, aucs_1k.get(name)),
            xlim=xlim, ylim=ylim,
            s_sel=12,
        )

    fig.legend(
        handles=legend_patches(), loc="lower center", ncol=2,
        fontsize=10, bbox_to_anchor=(0.5, -0.10),
    )
    fig.suptitle(
        "Bank-Marketing: Sampling Strategy Comparison  (Budget = 1,000)",
        fontsize=14, y=1.03,
    )
    fig.tight_layout()
    path = FIGURES_DIR / "bank_marketing_sampling_1k.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path.relative_to(RESULTS_DIR.parent)}")


# ── Figure B ──────────────────────────────────────────────────────────────────

def figure_b(Z_pool, y_pool, samples_scaling, maj_label, min_label, xlim, ylim, aucs_scaling):
    """Stratified vs Per-Class k-Center at 4 budgets — 2×4 grid."""
    strategies = ["stratified", "stratified_coreset"]
    row_labels  = ["Stratified", "Per-Class k-Center"]

    fig, axes = plt.subplots(2, 4, figsize=(22, 8.5))

    for row, (strat_name, row_label) in enumerate(zip(strategies, row_labels)):
        for col, budget in enumerate(BUDGETS):
            ax = axes[row, col]
            idx   = samples_scaling[(strat_name, budget)]
            Z_sel = Z_pool[idx]
            y_sel = y_pool[idx]
            auc   = aucs_scaling.get(strat_name, {}).get(budget)

            # Point size: 15 at 1K → 8 at 10K (linear)
            s_sel = 15.0 - (budget - 1000) / (10000 - 1000) * (15.0 - 8.0)

            draw_panel(
                ax, Z_pool, y_pool, Z_sel, y_sel, maj_label, min_label,
                title=f"Budget = {budget:,}",
                sub=subtitle_text(y_sel, maj_label, min_label, auc),
                xlim=xlim, ylim=ylim,
                s_sel=s_sel,
            )

            # Row label on the leftmost panel
            if col == 0:
                ax.set_ylabel(row_label, fontsize=12, fontweight="bold", labelpad=10)

    fig.legend(
        handles=legend_patches(), loc="lower center", ncol=2,
        fontsize=11, bbox_to_anchor=(0.5, -0.04),
    )
    fig.suptitle(
        "Bank-Marketing: Budget Scaling Comparison",
        fontsize=14, y=1.02,
    )
    fig.tight_layout()
    path = FIGURES_DIR / "bank_marketing_budget_scaling.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path.relative_to(RESULTS_DIR.parent)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Loading AUC annotations from Experiment 2 results ...")
    aucs_1k, aucs_scaling = _load_exp2_aucs(RESULTS_DIR, dataset="bank-marketing", seed=1)
    print(f"  1K AUCs loaded for strategies: {list(aucs_1k.keys())}")

    print("\nLoading bank-marketing ...")
    X_pool, y_pool = load_bank_marketing()
    print(f"Pool shape: {X_pool.shape}  class counts: {np.bincount(y_pool)}")

    maj_label, min_label = detect_majority(y_pool)
    print(f"Majority class: {maj_label}   Minority class: {min_label}")

    print("\nFitting PCA(2) on full training pool ...")
    Z_pool = fit_pca2(X_pool)
    xlim, ylim = shared_limits(Z_pool)

    # ── Figure A: all 5 strategies @ 1K ───────────────────────────────────────
    print("\nSampling — Figure A (all 5 strategies @ 1K, seed=1) ...")
    samples_1k = {}
    for name, sampler in SAMPLERS.items():
        print(f"  {name:<20s}", end="", flush=True)
        idx = sampler.sample(X_pool, y_pool, target_size=1000, seed=1)
        samples_1k[name] = idx
        print(f"  → {len(idx):,} points selected")

    print("\nGenerating Figure A ...")
    figure_a(Z_pool, y_pool, samples_1k, maj_label, min_label, xlim, ylim, aucs_1k)

    # ── Figure B: stratified + strat_coreset @ 4 budgets ──────────────────────
    print("\nSampling — Figure B (stratified & stratified_coreset @ 4 budgets, seed=1) ...")
    samples_scaling = {}
    for name in ["stratified", "stratified_coreset"]:
        sampler = SAMPLERS[name]
        for budget in BUDGETS:
            print(f"  {name:<20s} budget={budget:,}", end="", flush=True)
            idx = sampler.sample(X_pool, y_pool, target_size=budget, seed=1)
            samples_scaling[(name, budget)] = idx
            print(f"  → {len(idx):,} points selected")

    print("\nGenerating Figure B ...")
    figure_b(Z_pool, y_pool, samples_scaling, maj_label, min_label, xlim, ylim, aucs_scaling)

    print("\n" + "=" * 60)
    print("All visualisation figures saved.")
    print("=" * 60)


if __name__ == "__main__":
    run()
