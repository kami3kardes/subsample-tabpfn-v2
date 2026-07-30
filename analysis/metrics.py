import numpy as np
from sklearn.metrics import roc_auc_score


def compute_auc(y_test: np.ndarray, proba: np.ndarray, n_classes: int) -> float:
    """
    Compute AUC-ROC.

    Binary   → roc_auc_score(y_test, proba[:, 1])
    Multiclass → roc_auc_score(y_test, proba, multi_class='ovr')
    """
    if n_classes == 2:
        return float(roc_auc_score(y_test, proba[:, 1]))
    else:
        return float(roc_auc_score(y_test, proba, multi_class="ovr"))
