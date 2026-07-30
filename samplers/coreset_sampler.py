import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit
from samplers.base import BaseSampler
from configs.config import POOL_MAX_SIZE, PCA_DIMS


class CoresetSampler(BaseSampler):
    """
    Strategy 3: k-Center Greedy coreset selection.

    Selects a subset that minimises the maximum distance from any pool
    point to its nearest selected centre.

    For pools exceeding 500K rows, a stratified pre-filter is applied
    before coreset selection to maintain tractable computation.

    Steps:
      1. Stratified pre-filter if pool > POOL_MAX_SIZE
      2. StandardScaler normalisation
      3. PCA to 50 dims if n_features > 50  (critical for large datasets)
      4. Seed-controlled random starting point
      5. Greedy: iteratively add the point with max min-distance to selected set
      6. Batched distance computation for memory efficiency
    """

    def __init__(self, batch_size: int = 10_000):
        self.batch_size = batch_size

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
            index_map = pre_idx  # maps reduced indices → original indices
            X_pool = X_pool[pre_idx]
            y_pool = y_pool[pre_idx]
            n = len(X_pool)

        # 1. Normalise
        X = StandardScaler().fit_transform(X_pool.astype(np.float32))

        # 2. PCA projection
        if X.shape[1] > PCA_DIMS:
            n_components = min(PCA_DIMS, X.shape[0] - 1, X.shape[1])
            X = PCA(n_components=n_components, random_state=seed).fit_transform(X)

        X = X.astype(np.float32)

        # 3. Random start
        rng = np.random.RandomState(seed)
        first = rng.randint(0, n)
        selected = [first]

        # 4. Initialise min-distances to the first centre
        min_dists = self._batch_sq_distances(X, X[first])

        # 5. Greedy selection
        for i in range(1, target_size):
            new_idx = int(np.argmax(min_dists))
            selected.append(new_idx)

            # Update min-distances with new centre
            new_dists = self._batch_sq_distances(X, X[new_idx])
            np.minimum(min_dists, new_dists, out=min_dists)
            min_dists[new_idx] = -1.0  # mark as selected

            if i % 1000 == 0:
                print(f"    [coreset] {i}/{target_size} centres selected ...")

        selected = np.array(selected, dtype=np.int64)
        if index_map is not None:
            selected = index_map[selected]
        return selected

    def _batch_sq_distances(self, X: np.ndarray, centre: np.ndarray) -> np.ndarray:
        """Compute squared Euclidean distances from all rows in X to a single centre."""
        n = len(X)
        dists = np.empty(n, dtype=np.float32)
        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            diff = X[start:end] - centre
            dists[start:end] = np.sum(diff ** 2, axis=1)
        return dists
