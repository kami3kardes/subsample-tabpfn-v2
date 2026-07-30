import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from samplers.base import BaseSampler


class StratifiedSampler(BaseSampler):
    """Strategy 2: Stratified sampling — preserves class proportions."""

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

        ratio = target_size / n
        # StratifiedShuffleSplit "test" split gives us the selected subset
        sss = StratifiedShuffleSplit(n_splits=1, test_size=ratio, random_state=seed)
        _, selected = next(sss.split(X_pool, y_pool))
        return selected
