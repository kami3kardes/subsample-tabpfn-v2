"""
Illustrative figure: how each of our five samplers behaves on a 2D toy.

Uses the *actual* sampler implementations from samplers/ (not simplified
re-implementations), so the figure faithfully shows the same algorithms
that produced the benchmark numbers. Notably:

  - "Prototype" here means our nearest-enemy ranking + top-2k + uniform
    draw, NOT condensed-NN. (CNN is the Discarded Condensed Nearest
    Neighbour Rule of Section 4.2.6 — explicitly rejected as too
    aggressive on large pools.)
  - The distance-based samplers (k-Center, Prototype, Per-Class k-Center)
    StandardScaler-normalise features internally; on 2D toy data this
    just rescales axes proportionally, so the visual geometry is
    preserved.

Output: results/figures/sampler_behavior_demo.png
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

    # ── 1. Toy pool: 3 overlapping Gaussian blobs ─────────────────────
    # Centres chosen so class 0 and class 2 partially overlap on the
    # y-axis, creating a clear nearest-enemy boundary the Prototype
    # sampler should latch onto.
    X, y = make_blobs(
        n_samples=300,
        centers=[(0, 0), (5, 5), (0, 6)],
        cluster_std=[1.0, 1.0, 1.0],
        random_state=SEED,
    )
    n_sample = 60

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
        print(f"  {name:20s}: {len(idx)} points, class balance {balance}")

    # ── 3. Plot — 5 panels, one per sampler ───────────────────────────
    class_colors = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c"}
    class_names  = {0: "class 0 — (0, 0)",
                    1: "class 1 — (5, 5)",
                    2: "class 2 — (0, 6)"}

    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5))

    for ax, (name, idx) in zip(axes, selections.items()):
        # Background: full pool with low alpha
        for c in np.unique(y):
            m = y == c
            ax.scatter(X[m, 0], X[m, 1],
                       c=class_colors[c], alpha=0.18, s=22,
                       edgecolors="none", zorder=1)

        # Foreground: selected points with black edge
        sel_mask = np.zeros(len(X), dtype=bool)
        sel_mask[idx] = True
        for c in np.unique(y):
            csel = sel_mask & (y == c)
            ax.scatter(X[csel, 0], X[csel, 1],
                       c=class_colors[c], s=65,
                       edgecolor="black", linewidth=0.9, zorder=3)

        # Class-balance annotation in the title
        classes, counts = np.unique(y[idx], return_counts=True)
        balance = " / ".join(str(c) for c in counts)
        ax.set_title(f"{name}\n($k = {len(idx)}$, class counts: {balance})",
                     fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2, zorder=0)

    # Legend (place once, outside the rightmost axis)
    handles = [plt.Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=class_colors[c],
                          markeredgecolor="black", markeredgewidth=0.9,
                          markersize=9, label=class_names[c])
               for c in [0, 1, 2]]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               fontsize=10, bbox_to_anchor=(0.5, -0.04))

    pool_pct = round(100 * len(y) / len(y))   # placeholder
    fig.suptitle(
        "Sampling Strategy Behaviour on a 3-blob 2D Toy\n"
        f"Pool n = {len(y)}, sample k = {n_sample}, seed = {SEED}. "
        "Faded dots = pool; bold dots = selection. "
        "Distance-based samplers internally Standard-scale the 2D features.",
        fontsize=12, y=1.05,
    )

    out = FIGURES_DIR / "sampler_behavior_demo.png"
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
