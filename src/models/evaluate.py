"""
Evaluation module — business-relevant metrics beyond vanilla accuracy.
"""

import numpy as np
import pandas as pd
from typing import Dict
from sklearn.metrics import (
    average_precision_score,
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
    precision_recall_curve,
)


def fbeta_score(y_true, y_pred, beta: float = 2.0) -> float:
    """F-beta score. Beta=2 weights recall twice as much as precision."""
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    if p + r == 0:
        return 0.0
    return (1 + beta**2) * p * r / (beta**2 * p + r)


def evaluate_model(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
) -> Dict[str, float]:
    """
    Compute all evaluation metrics.
    Returns a dict ready to log to MLflow.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "auc_pr": float(average_precision_score(y_true, y_prob)),
        "auc_roc": float(roc_auc_score(y_true, y_prob)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "f2_score": float(fbeta_score(y_true, y_pred, beta=2.0)),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
    }


def compute_business_metric(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    avg_slot_value_usd: float = 120.0,
    overbooking_cost_usd: float = 50.0,
    daily_appointments: int = 200,
    noshow_rate: float = 0.20,
) -> Dict[str, float]:
    """
    Estimate real-world revenue impact.

    Logic:
    - True Positives: caught no-shows → slot can be given to another patient → revenue saved
    - False Positives: flagged but they showed up → intervention cost (unnecessary SMS/call)
    - False Negatives: missed no-shows → lost revenue
    - True Negatives: correctly predicted show-ups → no action needed

    Scale to full daily volume.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total = len(y_true)

    # Scale rates to daily appointment volume
    tp_rate = tp / total
    fp_rate = fp / total
    fn_rate = fn / total

    daily_tp = tp_rate * daily_appointments
    daily_fp = fp_rate * daily_appointments
    daily_fn = fn_rate * daily_appointments

    revenue_saved = daily_tp * avg_slot_value_usd
    intervention_cost = (daily_tp + daily_fp) * 2.0  # SMS/call cost ~$2
    overbooking_loss = daily_fp * overbooking_cost_usd * 0.1  # Only ~10% FP → actual overbook
    revenue_lost = daily_fn * avg_slot_value_usd

    net_value = revenue_saved - intervention_cost - overbooking_loss

    return {
        "estimated_revenue_saved_per_day": float(revenue_saved),
        "estimated_net_value_per_day": float(net_value),
        "estimated_revenue_lost_per_day": float(revenue_lost),
        "intervention_cost_per_day": float(intervention_cost),
    }


def get_optimal_threshold(y_true, y_prob) -> float:
    """
    Find the threshold that maximizes F2 score.
    Use this instead of default 0.5.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f2_scores = []
    for p, r in zip(precisions, recalls):
        if p + r == 0:
            f2_scores.append(0.0)
        else:
            f2 = (1 + 4) * p * r / (4 * p + r)
            f2_scores.append(f2)
    best_idx = int(np.argmax(f2_scores))
    return float(thresholds[min(best_idx, len(thresholds) - 1)])
