import numpy as np
from samplers.base import BaseSampler
from samplers.coreset_sampler import CoresetSampler


class StratifiedCoreset(BaseSampler):
    """
    Strategy 5: Stratified k-Center (hybrid).

    Splits the pool by class, allocates the budget proportionally,
    and runs k-Center independently within each class.
    This preserves class balance while maintaining geometric diversity.
    """

    def __init__(self, batch_size: int = 10_000):
        self._coreset = CoresetSampler(batch_size=batch_size)

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

        classes, counts = np.unique(y_pool, return_counts=True)
        fractions = counts / n

        selected = []
        allocated = 0

        for i, (c, frac) in enumerate(zip(classes, fractions)):
            class_idx = np.where(y_pool == c)[0]

            # Last class gets remaining budget to avoid rounding shortfall
            if i == len(classes) - 1:
                budget = target_size - allocated
            else:
                budget = max(1, round(target_size * frac))

            budget = min(budget, len(class_idx))
            allocated += budget

            # Run coreset within this class
            sub_idx = self._coreset.sample(
                X_pool[class_idx],
                y_pool[class_idx],
                target_size=budget,
                seed=seed,
            )
            selected.append(class_idx[sub_idx])

        return np.concatenate(selected)
