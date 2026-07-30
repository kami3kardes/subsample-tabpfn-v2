"""
Pareto-style scatter: mean AUC vs mean total time per strategy.

Each strategy is a single point in (time, AUC) space. The Pareto
front is the set of points not dominated by any other (i.e. no
other strategy is both faster AND more accurate).

Output: results/figures/pareto_auc_vs_time.png
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

COLORS = {
    "Random":              "#1f77b4",
    "Stratified":          "#ff7f0e",
    "k-Center":            "#2ca02c",
    "Prototype (NE)":      "#d62728",
    "Per-Class k-Center":  "#9467bd",
}


def main():
    df = pd.read_csv(RESULTS_DIR / "experiment_4_auc_vs_time.csv")
    df = df.drop_duplicates(subset=["strategy_label"]).reset_index(drop=True)
    print("Loaded:")
    print(df)

    # Compute Pareto front: a strategy is Pareto-optimal if no other
    # strategy has BOTH higher AUC AND lower time.
    on_front = []
    for i, row in df.iterrows():
        dominated = False
        for j, other in df.iterrows():
            if i == j:
                continue
            if (other["Mean AUC"] >= row["Mean AUC"]
                    and other["Mean Total Time (s)"] <= row["Mean Total Time (s)"]
                    and (other["Mean AUC"] > row["Mean AUC"]
                         or other["Mean Total Time (s)"] < row["Mean Total Time (s)"])):
                dominated = True
                break
        on_front.append(not dominated)
    df["pareto"] = on_front
    print("\nPareto-optimal strategies:")
    print(df[df["pareto"]][["strategy_label", "Mean AUC", "Mean Total Time (s)"]])

    # Plot
    fig, ax = plt.subplots(figsize=(9, 6))

    # Connect Pareto-front points with a line (sorted by time)
    pareto_df = df[df["pareto"]].sort_values("Mean Total Time (s)")
    ax.plot(
        pareto_df["Mean Total Time (s)"], pareto_df["Mean AUC"],
        color="black", linewidth=1.5, alpha=0.4, linestyle="--",
        zorder=1, label="Pareto front",
    )

    # Plot all 5 strategies
    for _, row in df.iterrows():
        name = row["strategy_label"]
        x, y = row["Mean Total Time (s)"], row["Mean AUC"]
        c = COLORS.get(name, "gray")
        marker = "o" if row["pareto"] else "X"
        size = 380 if row["pareto"] else 220
        edge = "black" if row["pareto"] else "darkred"
        edge_w = 1.8 if row["pareto"] else 2.4
        ax.scatter([x], [y], s=size, c=c, marker=marker,
                   edgecolor=edge, linewidth=edge_w, zorder=3,
                   label=name + (" ✓ Pareto" if row["pareto"] else " ✗ dominated"))
        # Annotate
        offset = (10, 8) if name not in ["Stratified", "Per-Class k-Center"] else (10, -20)
        ax.annotate(
            name,
            xy=(x, y), xytext=offset, textcoords="offset points",
            fontsize=11, fontweight="bold", color=c,
        )

    # Highlight the dominated region for Prototype
    proto_row = df[df["strategy_label"] == "Prototype (NE)"].iloc[0]
    rand_row = df[df["strategy_label"] == "Random"].iloc[0]
    # Shade rectangle: from Random's position to upper-right of Prototype
    from matplotlib.patches import Rectangle
    rect = Rectangle(
        (rand_row["Mean Total Time (s)"], df["Mean AUC"].min() - 0.01),
        proto_row["Mean Total Time (s)"] - rand_row["Mean Total Time (s)"],
        rand_row["Mean AUC"] - df["Mean AUC"].min() + 0.01,
        facecolor="lightcoral", alpha=0.12, zorder=0,
    )
    ax.add_patch(rect)
    ax.text(
        (rand_row["Mean Total Time (s)"] + proto_row["Mean Total Time (s)"]) / 2,
        df["Mean AUC"].min() + 0.001,
        "Dominated region\n(slower AND worse than Random)",
        ha="center", va="bottom", fontsize=9, color="darkred",
        style="italic", alpha=0.8,
    )

    ax.set_xlabel("Mean total time per run (seconds)  →  lower is better",
                  fontsize=12)
    ax.set_ylabel("Mean AUC  →  higher is better", fontsize=12)
    ax.set_title(
        "Strategy Pareto Front:  Quality vs Wall-Clock Cost\n"
        "Circles = Pareto-optimal     |     ✗ markers = dominated     |     "
        "Dashed line = Pareto front",
        fontsize=12, pad=12,
    )
    ax.grid(True, alpha=0.3, zorder=0)
    ax.set_xlim(140, 410)
    ax.set_ylim(0.862, 0.890)
    ax.legend(fontsize=9, loc="lower right", framealpha=0.92)

    fig.tight_layout()
    out = FIGURES_DIR / "pareto_auc_vs_time.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
