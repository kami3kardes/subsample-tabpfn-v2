import numpy as np
import pandas as pd
import openml
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

from configs.config import DATASETS


def load_dataset(dataset_id: int, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Fetch one dataset from OpenML and return (X, y) as numpy arrays."""
    print(f"\nLoading '{name}' (OpenML id={dataset_id}) ...")
    ds = openml.datasets.get_dataset(
        dataset_id,
        download_data=True,
        download_qualities=False,
        download_features_meta_data=False,
    )
    target_attr = ds.default_target_attribute
    X_df, y_series, _, _ = ds.get_data(target=target_attr)

    # Ordinal-encode categorical columns (string → integer)
    cat_cols = X_df.select_dtypes(include=["category", "object"]).columns
    if len(cat_cols) > 0:
        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_df[cat_cols] = oe.fit_transform(X_df[cat_cols])

    # Convert remaining columns to numeric; fill NaN with column median
    X_df = X_df.apply(pd.to_numeric, errors="coerce")
    medians = X_df.median(numeric_only=True)
    X_df = X_df.fillna(medians)
    # Drop any columns still all-NaN (safety net)
    X_df = X_df.dropna(axis=1, how="all")

    # Encode target to integer labels
    le = LabelEncoder()
    y = le.fit_transform(y_series.astype(str))

    X = X_df.values.astype(np.float32)

    n_classes = len(np.unique(y))
    counts = np.bincount(y)
    print(
        f"  rows={X.shape[0]:,}  features={X.shape[1]}  "
        f"classes={n_classes}  dist={dict(zip(range(n_classes), counts.tolist()))}"
    )
    return X, y


def load_all_datasets() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load all datasets defined in config and return {name: (X, y)}."""
    results = {}
    for name, did in DATASETS.items():
        try:
            X, y = load_dataset(did, name)
            results[name] = (X, y)
        except Exception as exc:
            print(f"  [ERROR] Failed to load '{name}': {exc}")
    return results
