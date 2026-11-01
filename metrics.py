"""
The seven multi-label metrics used in the CorGCN paper (Sec 5.1.4):
Ranking Loss, Hamming Loss, Macro-AUC, Micro-AUC, Macro-AP, Micro-AP,
Label Ranking AP (LRAP).
"""
import numpy as np
from sklearn.metrics import (
    label_ranking_loss, hamming_loss, roc_auc_score,
    average_precision_score, label_ranking_average_precision_score,
)


def evaluate(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5):
    """y_true: (n, K) 0/1. y_score: (n, K) probabilities in [0,1]."""
    y_pred = (y_score >= threshold).astype(int)

    # guard against classes with only one label value present (AUC undefined)
    valid_cols = [k for k in range(y_true.shape[1])
                  if len(np.unique(y_true[:, k])) > 1]

    results = {}
    results["ranking_loss"] = label_ranking_loss(y_true, y_score)
    results["hamming_loss"] = hamming_loss(y_true, y_pred)

    if valid_cols:
        results["macro_auc"] = roc_auc_score(
            y_true[:, valid_cols], y_score[:, valid_cols], average="macro")
        results["macro_ap"] = average_precision_score(
            y_true[:, valid_cols], y_score[:, valid_cols], average="macro")
    else:
        results["macro_auc"] = float("nan")
        results["macro_ap"] = float("nan")

    results["micro_auc"] = roc_auc_score(y_true, y_score, average="micro")
    results["micro_ap"] = average_precision_score(y_true, y_score, average="micro")
    results["lrap"] = label_ranking_average_precision_score(y_true, y_score)
    return results


def format_metrics(results: dict) -> str:
    return (
        f"Ranking={results['ranking_loss']*100:.2f} "
        f"Hamming={results['hamming_loss']*100:.2f} "
        f"Ma-AUC={results['macro_auc']*100:.2f} "
        f"Mi-AUC={results['micro_auc']*100:.2f} "
        f"Ma-AP={results['macro_ap']*100:.2f} "
        f"Mi-AP={results['micro_ap']*100:.2f} "
        f"LRAP={results['lrap']*100:.2f}"
    )
