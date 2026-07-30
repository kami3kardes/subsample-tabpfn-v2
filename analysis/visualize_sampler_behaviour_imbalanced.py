"""
Companion to visualize_sampler_behaviour.py — an *imbalanced* toy
with classes of unequal size AND different spreads. This setting
makes the differences between samplers visually dramatic:

  - Stratified vs Random: imbalance makes proportional rounding visible.
  - k-Center vs Per-Class k-Center: cluster_std varies (1.6 / 0.8 / 0.3),
    so the wide-spread class dominates vanilla k-Center's "spread points
    everywhere" objective and the tight class becomes under-represented.
  - Prototype (NE): boundary focus is sharper because two classes
    (0 and 2) overlap on the y-axis while class 1 sits diagonally.

Pool n = 6,000, sample k = 600 (10 % rate — matches our Exp 2 low-budget
condition). Uses the actual sampler implementations from samplers/, not
simplified rewrites; in particular "Prototype" is nearest-enemy ranking,
not CNN (which is the Discarded Condensed Nearest Neighbour Rule of
Section 4.2.6).

Output: results/figures/sampler_behavior_imbalanced.png
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.datasets import make_blobs

from samplers.random_sampler   import RandomSampler
from samplers.stratified_sampler import StratifiedSampler
from samplers.coreset_sampler  import CoresetSampler
from samplers.prototype_sampler import PrototypeSampler
from samplers.stratified_coreset import StratifiedCoreset


FIGURES_DIR = Path(__file__).parent.parent / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def main():
    SEED = 42
    np.random.seed(SEED)

    # ── 1. Imbalanced toy pool ────────────────────────────────────────
    # Class 0: 3,000 wide-spread points (std=1.6)  — dominant majority
    # Class 1: 2,000 medium-spread points (std=0.8) — middle
    # Class 2: 1,000 tight points (std=0.3)        — sparse minority
    # The 50/33/17 % split + varying density is what makes vanilla
    # k-Center skew its picks toward the wide majority class.
    X, y = make_blobs(
        n_samples=[3000, 2000, 1000],
        centers=[(0, 0), (5, 5), (0, 6)],
        cluster_std=[1.6, 0.8, 0.3],
        random_state=SEED,
    )
    n_total  = len(X)
    n_sample = 600

    pool_balance = np.bincount(y)
    print(f"Pool composition: {n_total} total — "
          f"class 0={pool_balance[0]} ({pool_balance[0]/n_total:.0%}), "
          f"class 1={pool_balance[1]} ({pool_balance[1]/n_total:.0%}), "
          f"class 2={pool_balance[2]} ({pool_balance[2]/n_total:.0%})\n")

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
        idx = sampler.sample(X.astype(np.float32), y, n_sample, SEED)
        selections[name] = idx
        classes, counts = np.unique(y[idx], return_counts=True)
        balance = "/".join(str(c) for c in counts)
        pct = "/".join(f"{c/n_sample:.0%}" for c in counts)
        print(f"  {name:20s}: k = {len(idx)}, class counts {balance} ({pct})")

    # ── 3. Six-panel plot: full pool + each sampler ───────────────────
    class_colors = {0: "#2196F3", 1: "#FF5722", 2: "#4CAF50"}
    class_names  = {0: f"class 0 — wide, n={pool_balance[0]} (std=1.6)",
                    1: f"class 1 — medium, n={pool_balance[1]} (std=0.8)",
                    2: f"class 2 — tight, n={pool_balance[2]} (std=0.3)"}

    fig, axes = plt.subplots(1, 6, figsize=(22, 4))

    # Panel 0: full pool
    color_map = np.array([class_colors[yi] for yi in y])
    axes[0].scatter(X[:, 0], X[:, 1], c=color_map, alpha=0.55, s=12,
                    edgecolors="none")
    axes[0].set_title(f"Full pool (n = {n_total})\n"
                      f"class counts {pool_balance[0]}/{pool_balance[1]}/{pool_balance[2]}",
                      fontsize=11, fontweight="bold")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].set_aspect("equal")
    axes[0].grid(True, alpha=0.2, zorder=0)

    # Panels 1-5: samplers
    for ax, (name, idx) in zip(axes[1:], selections.items()):
        # Background: full pool at low alpha
        ax.scatter(X[:, 0], X[:, 1], c=color_map, alpha=0.08, s=10,
                   edgecolors="none", zorder=1)
        # Foreground: selected points
        ax.scatter(X[idx, 0], X[idx, 1], c=color_map[idx], s=40,
                   edgecolor="black", linewidth=0.7, zorder=3)

        classes, counts = np.unique(y[idx], return_counts=True)
        # Pad to 3 classes for consistent reporting (in case a sampler
        # somehow drops one — informative either way).
        full_counts = [int(np.sum(y[idx] == c)) for c in [0, 1, 2]]
        ax.set_title(f"{name}\n"
                     f"($k = {len(idx)}$, "
                     f"class counts {full_counts[0]}/{full_counts[1]}/{full_counts[2]})",
                     fontsize=11, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2, zorder=0)

    # Shared legend at the bottom
    handles = [plt.Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=class_colors[c],
                          markeredgecolor="black", markeredgewidth=0.7,
                          markersize=9, label=class_names[c])
               for c in [0, 1, 2]]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               fontsize=10, bbox_to_anchor=(0.5, -0.04))

    fig.suptitle(
        f"Sampling Strategy Behaviour on an Imbalanced 3-blob Toy\n"
        f"Pool n = {n_total}, sample k = {n_sample} (10 % rate), seed = {SEED}.  "
        "Faded dots = pool; bold dots = selection.\n"
        "Pool class balance: 50 / 33 / 17 %.  "
        "Cluster std: 1.6 / 0.8 / 0.3 (wide majority → tight minority).",
        fontsize=12, y=1.06,
    )

    out = FIGURES_DIR / "sampler_behavior_imbalanced.png"
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
