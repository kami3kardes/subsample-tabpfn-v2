"""
Candidate-composition diagnostic for the nearest-enemy Prototype sampler,
plus the within-class geometry diagnostic for Per-Class k-Center.

Discharges the future-work measurement promised in Chapter 1, Contribution 3:
  (a) per-class composition of the Prototype sampler's top-2k nearest-enemy
      candidate set (and of the selected k) on covertype and connect-4,
      together with the nearest-enemy pair structure — testing the
      majority-boundary-dominance mechanism of Chapter 5, Section 5.5 and
      the class-2-boundary mechanism of Chapter 6, Section 6.3;
  (b) the within-class centroid-distance percentile of Per-Class k-Center
      selections on covertype versus a stratified (class-count-matched)
      draw — testing the peripheral-within-class mechanism of Chapter 5,
      Finding 3 and Chapter 8, Section 8.3.

Sampler-only: no TabPFN calls. Replays the exact preprocessing of
PrototypeSampler / StratifiedCoreset (StandardScaler -> PCA-50 when
n_features > PCA_DIMS) on the exact Experiment-1 pool (80/20 stratified
split, SPLIT_RANDOM_STATE).

The nearest-enemy distances are seed-invariant: connect-4 (42 features)
takes no PCA, and covertype's PCA at n_components = 50 >= 0.8 * 54
resolves to the deterministic full-SVD solver, which ignores
random_state. The script verifies this empirically (seed 1 vs seed 2
transforms) before reusing distances across seeds; only the final
uniform draw from the candidate set varies with the seed.

Outputs:
  results/candidate_composition.csv          (per dataset/budget/class rows)
  results/candidate_ne_pairs.csv             (nearest-enemy pair matrix)
  results/perclass_geometry_covertype.csv    (PC k-Center vs stratified)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import BallTree
from sklearn.preprocessing import StandardScaler

from configs.config import DATASETS, PCA_DIMS, SPLIT_RANDOM_STATE, TEST_SIZE
from preprocessing.data_loader import load_dataset
from samplers.stratified_coreset import StratifiedCoreset
from samplers.stratified_sampler import StratifiedSampler

RESULTS_DIR = Path(__file__).parent.parent / "results"
SEEDS = [1, 2, 3, 4]
TARGET_SIZES = [10_000, 1_000]  # full budget and frac = 0.10


def load_pool(name):
    X, y = load_dataset(DATASETS[name], name)
    X_pool, _, y_pool, _ = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SPLIT_RANDOM_STATE
    )
    return X_pool, y_pool


def prototype_feature_space(X_pool, seed):
    """Exact replica of PrototypeSampler preprocessing (no pre-filter:
    every pool in the benchmark is below POOL_MAX_SIZE)."""
    X = StandardScaler().fit_transform(X_pool.astype(np.float32))
    if X.shape[1] > PCA_DIMS:
        n_components = min(PCA_DIMS, X.shape[0] - 1, X.shape[1])
        X = PCA(n_components=n_components, random_state=seed).fit_transform(X)
    return X.astype(np.float32)


def nearest_enemy(X, y):
    """Per-point nearest-enemy distance and the enemy's class label."""
    n = len(X)
    dist = np.full(n, np.inf)
    enemy_class = np.full(n, -1)
    for c in np.unique(y):
        mask_c = y == c
        mask_e = ~mask_c
        if mask_e.sum() == 0:
            continue
        tree = BallTree(X[mask_e])
        d, idx = tree.query(X[mask_c], k=1)
        dist[mask_c] = d.ravel()
        enemy_class[mask_c] = y[mask_e][idx.ravel()]
    return dist, enemy_class


def composition(y, idx, classes):
    frac = np.array([(y[idx] == c).mean() for c in classes])
    return frac


def run_prototype_diagnostic(name):
    print(f"\n{'=' * 70}\nPROTOTYPE CANDIDATE COMPOSITION — {name}\n{'=' * 70}")
    X_pool, y_pool = load_pool(name)
    classes = np.unique(y_pool)
    pool_frac = composition(y_pool, np.arange(len(y_pool)), classes)

    # Seed-invariance check of the feature space (hence of NE distances)
    X1 = prototype_feature_space(X_pool, seed=1)
    X2 = prototype_feature_space(X_pool, seed=2)
    invariant = np.allclose(np.abs(X1), np.abs(X2), atol=1e-4)
    print(f"feature space seed-invariant: {invariant}")

    dist, enemy_class = nearest_enemy(X1, y_pool)
    order = np.argsort(dist)

    comp_rows, pair_rows = [], []
    for target in TARGET_SIZES:
        n_cand = min(2 * target, len(X_pool))
        cand = order[:n_cand]
        cand_frac = composition(y_pool, cand, classes)

        # nearest-enemy pair matrix over the candidate set
        for c in classes:
            m = y_pool[cand] == c
            for e in classes:
                if e == c:
                    continue
                cnt = int((enemy_class[cand][m] == e).sum())
                if cnt:
                    pair_rows.append(
                        dict(dataset=name, target_size=target, own_class=int(c),
                             enemy_class=int(e), count=cnt,
                             frac_of_candidates=cnt / n_cand)
                    )

        # per-seed selected composition (uniform draw from candidates)
        sel_fracs = []
        for seed in SEEDS:
            if not invariant and seed > 1:
                Xs = prototype_feature_space(X_pool, seed=seed)
                ds, _ = nearest_enemy(Xs, y_pool)
                cand_s = np.argsort(ds)[:n_cand]
            else:
                cand_s = cand
            rng = np.random.RandomState(seed)
            chosen = rng.choice(len(cand_s), target, replace=False)
            sel_fracs.append(composition(y_pool, cand_s[chosen], classes))
        sel_fracs = np.array(sel_fracs)

        print(f"\n-- target_size = {target:,} (candidates = {n_cand:,}) --")
        print(f"{'class':>6} {'pool%':>8} {'cand%':>8} {'ratio':>7} "
              f"{'sel% (mean±std over seeds)':>28}")
        for j, c in enumerate(classes):
            ratio = cand_frac[j] / pool_frac[j] if pool_frac[j] > 0 else np.nan
            print(f"{c:>6} {pool_frac[j]*100:>7.2f}% {cand_frac[j]*100:>7.2f}% "
                  f"{ratio:>6.2f}x {sel_fracs[:, j].mean()*100:>10.2f}% ± "
                  f"{sel_fracs[:, j].std()*100:.2f}%")
            comp_rows.append(
                dict(dataset=name, target_size=target, class_label=int(c),
                     pool_frac=pool_frac[j], candidate_frac=cand_frac[j],
                     representation_ratio=ratio,
                     selected_frac_mean=sel_fracs[:, j].mean(),
                     selected_frac_std=sel_fracs[:, j].std())
            )
    return comp_rows, pair_rows


def run_perclass_geometry(name="covertype", target=10_000):
    """Mean within-class centroid-distance percentile of selected points,
    Per-Class k-Center vs Stratified, in the standardized PCA-50 pool space."""
    print(f"\n{'=' * 70}\nPER-CLASS k-CENTER GEOMETRY — {name}\n{'=' * 70}")
    X_pool, y_pool = load_pool(name)
    classes = np.unique(y_pool)
    X = prototype_feature_space(X_pool, seed=1)  # shared measurement space

    # per-point distance to own-class centroid, and within-class percentile
    pctl = np.empty(len(X))
    for c in classes:
        m = y_pool == c
        d = np.linalg.norm(X[m] - X[m].mean(axis=0), axis=1)
        ranks = d.argsort().argsort()
        pctl[m] = ranks / (m.sum() - 1) * 100

    rows = []
    for sampler, label in [(StratifiedCoreset(), "perclass_kcenter"),
                           (StratifiedSampler(), "stratified")]:
        for seed in SEEDS:
            sel = sampler.sample(X_pool, y_pool, target_size=target, seed=seed)
            for c in classes:
                m = y_pool[sel] == c
                if m.sum() == 0:
                    continue
                rows.append(dict(dataset=name, sampler=label, seed=seed,
                                 class_label=int(c), n_selected=int(m.sum()),
                                 mean_centroid_pctl=pctl[sel][m].mean()))
    df = pd.DataFrame(rows)
    summary = (df.groupby(["sampler", "class_label"])["mean_centroid_pctl"]
               .agg(["mean", "std"]).round(1))
    print("\nmean within-class centroid-distance percentile "
          "(uniform draw expectation = 50):")
    print(summary.to_string())
    overall = df.groupby(["sampler", "seed"]).apply(
        lambda g: np.average(g.mean_centroid_pctl, weights=g.n_selected),
        include_groups=False).groupby("sampler").agg(["mean", "std"]).round(1)
    print("\nselection-weighted overall percentile per sampler:")
    print(overall.to_string())
    return df


if __name__ == "__main__":
    all_comp, all_pairs = [], []
    for name in ["connect-4", "covertype"]:
        comp, pairs = run_prototype_diagnostic(name)
        all_comp += comp
        all_pairs += pairs
    pd.DataFrame(all_comp).to_csv(RESULTS_DIR / "candidate_composition.csv", index=False)
    pd.DataFrame(all_pairs).to_csv(RESULTS_DIR / "candidate_ne_pairs.csv", index=False)
    geo = run_perclass_geometry()
    geo.to_csv(RESULTS_DIR / "perclass_geometry_covertype.csv", index=False)
    print("\nSaved: candidate_composition.csv, candidate_ne_pairs.csv, "
          "perclass_geometry_covertype.csv")
