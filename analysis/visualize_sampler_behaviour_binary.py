"""
Binary-imbalanced toy where the minority class sits *between* two
majority subclusters. This is the cleanest possible illustration of
the failure mode that drives our headline finding:

  - Stratified preserves the 88 / 12 pool ratio.
  - The Prototype (NE) sampler inverts it: every minority point has a
    very small nearest-enemy distance (a majority neighbour is right
    next door), so the top-2k boundary candidate set is dominated by
    minority points. Drawing k uniformly from those candidates yields
    a sample where the minority is massively over-represented and the
    majority is starved.
  - k-Center skews the other way (towards the wide majority cluster)
    because peripheral majority points dominate the max-min objective.
  - Per-Class k-Center re-imposes the pool ratio AND adds within-class
    coverage.

This single figure shows visually why distributional fidelity matters
more than geometric coverage for TabPFN v2, and why the boundary-focused
samplers landed in the bottom cluster of every aggregate metric.

Uses the actual sampler implementations from samplers/, not simplified
rewrites; "Prototype" is nearest-enemy ranking, not CNN.

Output: results/figures/sampler_behavior_binary_imbalanced.png
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from samplers.random_sampler   import RandomSampler
from samplers.stratified_sampler import StratifiedSampler
from samplers.coreset_sampler  import CoresetSampler
from samplers.prototype_sampler import PrototypeSampler
from samplers.stratified_coreset import StratifiedCoreset


FIGURES_DIR = Path(__file__).parent.parent / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def main():
    SEED = 42
    rng = np.random.default_rng(SEED)

    # ── 1. Binary imbalanced toy ──────────────────────────────────────
    # Two majority subclusters at (0,0) and (5,0) → cluster_std 1.0/0.8.
    # The minority class is *embedded between them* at (2.5, 0) with
    # tight std 0.5: every minority point sits in the boundary region.
    majority_1 = rng.standard_normal((150, 2)) * 1.0 + np.array([0, 0])
    majority_2 = rng.standard_normal((100, 2)) * 0.8 + np.array([5, 0])
    minority   = rng.standard_normal((35, 2)) * 0.5 + np.array([2.5, 0])

    X = np.vstack([majority_1, majority_2, minority]).astype(np.float32)
    y = np.concatenate([np.zeros(250, dtype=int), np.ones(35, dtype=int)])
    n_total  = len(X)
    n_sample = 60

    pool_balance = np.bincount(y)
    maj_pct = 100 * pool_balance[0] / n_total
    min_pct = 100 * pool_balance[1] / n_total
    print(f"Pool composition: {n_total} total — "
          f"majority {pool_balance[0]} ({maj_pct:.0f}%), "
          f"minority {pool_balance[1]} ({min_pct:.0f}%)\n")

    # ── 2. Run each sampler ───────────────────────────────────────────
    samplers = {
        "Random":             RandomSampler(),
        "Stratified":         StratifiedSampler(),
        "k-Center":           CoresetSampler(),
        "Prototype (NE)":     PrototypeSampler(),
        "Per-Class k-Center": StratifiedCoreset(),
    }

    selections = {}
    for name, sampler in samplers.items():
        idx = sampler.sample(X, y, n_sample, SEED)
        selections[name] = idx
        n_maj = int(np.sum(y[idx] == 0))
        n_min = int(np.sum(y[idx] == 1))
        min_share = 100 * n_min / len(idx)
        print(f"  {name:20s}: k = {len(idx)},  "
              f"majority {n_maj} ({100-min_share:.0f}%), "
              f"minority {n_min} ({min_share:.0f}%)")

    # ── 3. Six-panel plot ─────────────────────────────────────────────
    colors = {0: "#2196F3", 1: "#FF5722"}
    color_all = np.array([colors[yi] for yi in y])

    fig, axes = plt.subplots(1, 6, figsize=(26, 4.5))

    # Panel 0: full pool
    axes[0].scatter(X[y == 0, 0], X[y == 0, 1],
                    c=colors[0], alpha=0.55, s=22, edgecolors="none",
                    label=f"Majority (n={pool_balance[0]})")
    axes[0].scatter(X[y == 1, 0], X[y == 1, 1],
                    c=colors[1], alpha=0.85, s=28, edgecolors="none",
                    label=f"Minority (n={pool_balance[1]})")
    axes[0].set_title(f"Full pool (n = {n_total})\n"
                      f"{maj_pct:.0f}% / {min_pct:.0f}%",
                      fontsize=11, fontweight="bold")
    axes[0].legend(fontsize=8.5, loc="upper right",
                   markerscale=1.2, framealpha=0.9)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].set_aspect("equal")
    axes[0].grid(True, alpha=0.2, zorder=0)

    # Panels 1-5: samplers
    for ax, (name, idx) in zip(axes[1:], selections.items()):
        n_maj = int(np.sum(y[idx] == 0))
        n_min = int(np.sum(y[idx] == 1))
        min_share = 100 * n_min / len(idx)

        # Background pool at very low alpha
        ax.scatter(X[:, 0], X[:, 1], c=color_all, alpha=0.07,
                   s=10, edgecolors="none", zorder=1)
        # Foreground selection
        ax.scatter(X[idx, 0], X[idx, 1], c=color_all[idx], s=48,
                   edgecolor="black", linewidth=0.7, zorder=3)

        ax.set_title(
            f"{name}\n"
            f"$k = {len(idx)}$  •  "
            f"{n_maj} maj / {n_min} min  •  {min_share:.0f}% minority",
            fontsize=10, fontweight="bold",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2, zorder=0)

    # Shared axis limits so all 6 panels are visually comparable
    pad_x = 0.5
    pad_y = 0.5
    xlim = (X[:, 0].min() - pad_x, X[:, 0].max() + pad_x)
    ylim = (X[:, 1].min() - pad_y, X[:, 1].max() + pad_y)
    for ax in axes:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    fig.suptitle(
        "Sampling Strategy Behaviour on an Imbalanced Binary Toy\n"
        f"Pool n = {n_total} (88 % majority / 12 % minority), "
        f"sample k = {n_sample}, seed = {SEED}.   "
        "Minority class sits between two majority subclusters — every "
        "minority point is a boundary point.",
        fontsize=12, y=1.06,
    )

    out = FIGURES_DIR / "sampler_behavior_binary_imbalanced.png"
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
