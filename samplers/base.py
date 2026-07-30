import numpy as np


class BaseSampler:
    """Abstract base class for all subsampling strategies."""

    def sample(
        self,
        X_pool: np.ndarray,
        y_pool: np.ndarray,
        target_size: int,
        seed: int,
    ) -> np.ndarray:
        """
        Select indices from the pool.

        Parameters
        ----------
        X_pool : (n, d) feature matrix of the training pool
        y_pool : (n,) label array of the training pool
        target_size : desired number of samples to select
        seed : random seed for reproducibility

        Returns
        -------
        indices : 1-D int array of selected row indices into X_pool / y_pool
        """
        raise NotImplementedError
