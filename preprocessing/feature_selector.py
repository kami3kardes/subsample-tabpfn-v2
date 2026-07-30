import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif


def select_features(X: np.ndarray, y: np.ndarray, max_features: int = 500) -> np.ndarray:
    """
    Two-step feature selection applied only when needed:
    1. Drop features with >0.95 pairwise absolute correlation (redundancy removal).
    2. If still > max_features, keep top-N by mutual information with target.

    Returns reduced X (same number of rows, fewer columns).
    """
    n_features = X.shape[1]
    if n_features <= max_features:
        return X  # nothing to do for most datasets

    print(f"  [feature_selector] {n_features} features → applying reduction ...")

    # Step 1: correlation-based redundancy removal
    df = pd.DataFrame(X)
    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
    keep_mask = [i for i in range(n_features) if i not in to_drop]
    X = X[:, keep_mask]
    print(f"  [feature_selector] After corr removal: {X.shape[1]} features")

    # Step 2: MI filtering if still above threshold
    if X.shape[1] > max_features:
        mi = mutual_info_classif(X, y, random_state=0)
        top_idx = np.argsort(mi)[::-1][:max_features]
        X = X[:, np.sort(top_idx)]
        print(f"  [feature_selector] After MI filtering: {X.shape[1]} features")

    return X
