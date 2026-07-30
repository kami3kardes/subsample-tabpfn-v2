"""
Bimodal-classes toy — the figure that exposes Per-Class k-Center's
"stable but wrong" failure mode that the single-Gaussian toys cannot show.

Setup: each class is a *mixture* of two sub-Gaussians at different
locations.

  Class 0:
    - sub-Gaussian A at (-3, 0), std 1.0  (200 pts, diffuse)
    - sub-Gaussian B at ( 3, 0), std 0.4  (200 pts, tight)
  Class 1:
    - sub-Gaussian C at ( 0,  3), std 0.7  (200 pts)
    - sub-Gaussian D at ( 0, -3), std 0.7  (200 pts)

Why this exposes Per-Class k-Center: max-min greedy *within* class 0
picks the most spread-out points of class 0, which lands on the
PERIPHERIES of A and B and ignores both sub-Gaussian INTERIORS. Same
for class 1 across C and D. Random and Stratified, by contrast,
preserve density — they pick predominantly from the high-density
interiors. This within-class distributional mismatch is the
mechanism behind Per-Class k-Center's 2.7× ECE blow-up in our
benchmark, and it cannot be revealed by single-Gaussian toys where
"interior" and "periphery" are the same shape at different scales.

Output: results/figures/sampler_behavior_bimodal.png
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from pathlib import Path

from samplers.random_sampler   import RandomSampler
from samplers.stratified_sampler import StratifiedSampler
from samplers.coreset_sampler  import CoresetSampler
from samplers.prototype_sampler import PrototypeSampler
from samplers.stratified_coreset import StratifiedCoreset


FIGURES_DIR = Path(__file__).parent.parent / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# 1σ contour parameters for the four sub-Gaussians (centre, std)
SUB_GAUSS = [
    {"centre": (-3,  0), "std": 1.0, "label": "A"},  # class 0
    {"centre": ( 3,  0), "std": 0.4, "label": "B"},  # class 0
    {"centre": ( 0,  3), "std": 0.7, "label": "C"},  # class 1
    {"centre": ( 0, -3), "std": 0.7, "label": "D"},  # class 1
]
SUB_GAUSS_CLASS = [0, 0, 1, 1]


def interior_share(X_picks, threshold_factor=1.0):
    """Fraction of picks within `threshold_factor` × std of any sub-Gaussian centre."""
    if len(X_picks) == 0:
        return 0.0
    inside = np.zeros(len(X_picks), dtype=bool)
    for sg in SUB_GAUSS:
        cx, cy = sg["centre"]
        s = sg["std"]
        d = np.sqrt((X_picks[:, 0] - cx) ** 2 + (X_picks[:, 1] - cy) ** 2)
        inside |= (d <= threshold_factor * s)
    return inside.mean()


def main():
    SEED = 42
    rng  = np.random.default_rng(SEED)

    # ── 1. Build the bimodal-classes pool ─────────────────────────────
    sg_pts = []
    for sg in SUB_GAUSS:
        cx, cy = sg["centre"]
        sg_pts.append(rng.standard_normal((200, 2)) * sg["std"] + np.array([cx, cy]))

    X = np.vstack(sg_pts).astype(np.float32)
    y = np.array(SUB_GAUSS_CLASS).repeat(200)
    n_total  = len(X)
    n_sample = 80

    pool_balance = np.bincount(y)
    print(f"Pool composition: {n_total} total, "
          f"class 0={pool_balance[0]}, class 1={pool_balance[1]}\n")
    print(f"Each class is a *bimodal mixture* of two sub-Gaussians "
          f"(class 0: A wide + B tight; class 1: C top + D bottom).\n")

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
        # Interior share (within 1σ of any sub-Gaussian centre)
        interior_pct = 100 * interior_share(X[idx], threshold_factor=1.0)
        # Per-sub-Gaussian distribution of picks
        per_sg = []
        for sg in SUB_GAUSS:
            cx, cy = sg["centre"]; s = sg["std"]
            d = np.sqrt((X[idx, 0] - cx) ** 2 + (X[idx, 1] - cy) ** 2)
            per_sg.append(int((d <= 2 * s).sum()))
        sg_str = "/".join(str(c) for c in per_sg)
        print(f"  {name:20s}: k = {len(idx)}, "
              f"interior 1σ fraction = {interior_pct:5.1f}%, "
              f"per-subcluster (A/B/C/D) = {sg_str}")

    # ── 3. Six-panel plot ─────────────────────────────────────────────
    class_colors = {0: "#2196F3", 1: "#FF5722"}
    color_all = np.array([class_colors[yi] for yi in y])

    fig, axes = plt.subplots(1, 6, figsize=(26, 4.7))

    def draw_sigma_circles(ax, lw=0.9, alpha=0.5):
        for sg in SUB_GAUSS:
            cx, cy = sg["centre"]
            s = sg["std"]
            cls_color = class_colors[SUB_GAUSS_CLASS[SUB_GAUSS.index(sg)]]
            ax.add_patch(Circle((cx, cy), s, fill=False,
                                edgecolor=cls_color, linestyle="--",
                                linewidth=lw, alpha=alpha, zorder=2))

    # Panel 0: full pool with 1σ circles annotated A/B/C/D
    axes[0].scatter(X[y == 0, 0], X[y == 0, 1],
                    c=class_colors[0], alpha=0.55, s=15, edgecolors="none")
    axes[0].scatter(X[y == 1, 0], X[y == 1, 1],
                    c=class_colors[1], alpha=0.55, s=15, edgecolors="none")
    draw_sigma_circles(axes[0], lw=1.2, alpha=0.7)
    for sg in SUB_GAUSS:
        cx, cy = sg["centre"]
        axes[0].annotate(sg["label"], (cx, cy),
                         fontsize=11, ha="center", va="center",
                         fontweight="bold", color="black",
                         bbox=dict(boxstyle="circle,pad=0.15",
                                   fc="white", ec="black", alpha=0.85),
                         zorder=4)
    axes[0].set_title(f"Full pool (n = {n_total})\n"
                      "class 0 = A ∪ B,  class 1 = C ∪ D",
                      fontsize=11, fontweight="bold")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].set_aspect("equal")
    axes[0].grid(True, alpha=0.2, zorder=0)

    # Panels 1-5: samplers
    for ax, (name, idx) in zip(axes[1:], selections.items()):
        # Background pool at low alpha
        ax.scatter(X[:, 0], X[:, 1], c=color_all, alpha=0.08,
                   s=10, edgecolors="none", zorder=1)
        # 1σ circles (faint)
        draw_sigma_circles(ax, lw=0.7, alpha=0.45)
        # Foreground selection
        ax.scatter(X[idx, 0], X[idx, 1], c=color_all[idx], s=52,
                   edgecolor="black", linewidth=0.7, zorder=3)

        interior_pct = 100 * interior_share(X[idx], threshold_factor=1.0)
        ax.set_title(
            f"{name}\n"
            f"$k = {len(idx)}$  •  "
            f"{interior_pct:.0f}% of picks inside 1σ",
            fontsize=10.5, fontweight="bold",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2, zorder=0)

    # Shared axis limits
    pad = 0.7
    xlim = (X[:, 0].min() - pad, X[:, 0].max() + pad)
    ylim = (X[:, 1].min() - pad, X[:, 1].max() + pad)
    for ax in axes:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    fig.suptitle(
        "Sampling Strategy Behaviour on a Bimodal-Classes Toy\n"
        f"Each class is a mixture of two sub-Gaussians.  "
        f"Pool n = {n_total}, sample k = {n_sample}, seed = {SEED}.   "
        "Dashed circles = 1σ of each sub-Gaussian centre.\n"
        "High '% inside 1σ' = picks land on dense INTERIORS (distributional fidelity).  "
        "Low '% inside 1σ' = picks land on sparse PERIPHERIES (geometric coverage at cost of fidelity).",
        fontsize=11.5, y=1.07,
    )

    out = FIGURES_DIR / "sampler_behavior_bimodal.png"
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
