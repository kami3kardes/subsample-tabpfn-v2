import numpy as np
from samplers.base import BaseSampler


class RandomSampler(BaseSampler):
    """Strategy 1: Uniform random sampling."""

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
        rng = np.random.RandomState(seed)
        return rng.choice(n, size=target_size, replace=False)
