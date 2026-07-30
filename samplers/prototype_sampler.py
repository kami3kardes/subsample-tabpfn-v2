import numpy as np
from sklearn.neighbors import BallTree
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit
from samplers.base import BaseSampler
from configs.config import POOL_MAX_SIZE, PCA_DIMS


class PrototypeSampler(BaseSampler):
    """Boundary-focused sampling via nearest-enemy distance.

    For each point, computes distance to its nearest enemy (closest point
    of a different class). Collects the top 2×target_size candidates
    (all near the decision boundary) then randomly subsamples target_size
    from them using the seed.  This gives:
      - Boundary focus: every selected point is in the boundary region
      - Seed-dependent variation: different seeds pick different subsets,
        enabling genuine diversity across Experiment 3 ensemble members

    Features are StandardScaler-normalised and (when n_features > PCA_DIMS)
    PCA-reduced before distance computation, identical to CoresetSampler.
    This keeps the two distance-based samplers on the same feature
    geometry so cross-strategy comparisons are interpretable.
    """

    def sample(
        self,
        X_pool: np.ndarray,
        y_pool: np.ndarray,
        target_size: int,
        seed: int,
    ) -> np.ndarray:
        n = len(X_pool)
        if n <= target_size:
            return np.arange(n)

        # Pre-filter large pools with stratified random sampling
        index_map = None
        if n > POOL_MAX_SIZE:
            ratio = POOL_MAX_SIZE / n
            sss = StratifiedShuffleSplit(n_splits=1, test_size=ratio, random_state=seed)
            _, pre_idx = next(sss.split(X_pool, y_pool))
            index_map = pre_idx
            X_pool = X_pool[pre_idx]
            y_pool = y_pool[pre_idx]
            n = len(X_pool)

        # Normalise so all features contribute equally to Euclidean distance
        X = StandardScaler().fit_transform(X_pool.astype(np.float32))

        # PCA projection — matches CoresetSampler's feature space so that
        # nearest-enemy distances and k-Center coreset distances are computed
        # in the same geometry. Critical for high-dimensional datasets where
        # Euclidean distances in the original space lose discriminative power.
        if X.shape[1] > PCA_DIMS:
            n_components = min(PCA_DIMS, X.shape[0] - 1, X.shape[1])
            X = PCA(n_components=n_components, random_state=seed).fit_transform(X)

        X = X.astype(np.float32)

        classes = np.unique(y_pool)
        enemy_distances = np.full(n, np.inf)

        for c in classes:
            mask_c     = (y_pool == c)
            mask_enemy = ~mask_c
            if mask_enemy.sum() == 0:
                continue
            tree = BallTree(X[mask_enemy])
            dists, _ = tree.query(X[mask_c], k=1)
            enemy_distances[mask_c] = dists.ravel()

        # Top 2× candidates — all near the decision boundary
        n_candidates = min(2 * target_size, n)
        candidate_idx = np.argsort(enemy_distances)[:n_candidates]

        # Randomly subsample from candidates using seed
        rng = np.random.RandomState(seed)
        chosen = rng.choice(len(candidate_idx), target_size, replace=False)
        selected = candidate_idx[chosen]

        if index_map is not None:
            selected = index_map[selected]
        return selected
