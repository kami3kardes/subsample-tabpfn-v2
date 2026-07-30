import pandas as pd
from scipy.stats import wilcoxon


def wilcoxon_pairwise(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pairwise Wilcoxon signed-rank tests between all strategy pairs.
    Used in Experiment 1 analysis.

    Parameters
    ----------
    results_df : DataFrame with columns [dataset, strategy, seed, auc]
                 (Experiment 1 format; Experiment 3 uses 'inner_strategy')

    Returns
    -------
    DataFrame with columns [strategy_a, strategy_b, statistic, p_value]
    """
    strategies = results_df["strategy"].unique()
    rows = []
    for i, a in enumerate(strategies):
        for b in strategies[i + 1 :]:
            aucs_a = results_df[results_df["strategy"] == a].set_index(
                ["dataset", "seed"]
            )["auc"]
            aucs_b = results_df[results_df["strategy"] == b].set_index(
                ["dataset", "seed"]
            )["auc"]
            common = aucs_a.index.intersection(aucs_b.index)
            if len(common) < 2:
                continue
            stat, p = wilcoxon(aucs_a[common].values, aucs_b[common].values)
            rows.append({"strategy_a": a, "strategy_b": b, "statistic": stat, "p_value": p})
    return pd.DataFrame(rows)
