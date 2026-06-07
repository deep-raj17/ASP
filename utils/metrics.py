"""
utils/metrics.py
────────────────────────────────────────────────────────
Full evaluation suite for anomaly detection:
  - Accuracy, Precision, Recall, F1
  - ROC-AUC, PR-AUC
  - Partial AUC (pAUC, default max_fpr=0.1)
  - Log Loss
  - Equal Error Rate (EER) from DET curve
  - Confusion Matrix
  - Pretty-print summary
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, asdict
from typing import List

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    log_loss,
    roc_curve,
    det_curve,
)


@dataclass
class EvalMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    p_auc: float           # partial AUC  (max_fpr = 0.1)
    log_loss: float
    eer: float
    threshold: float
    confusion_matrix: List[List[int]]
    # At fixed 0.5 threshold (deployment default); avoids "perfect" Youden-only stats
    accuracy_at_05: float = 0.0
    f1_at_05: float = 0.0
    balanced_accuracy: float = 0.0  # at Youden optimal threshold

    def to_dict(self) -> dict:
        return asdict(self)

    def pretty(self) -> str:
        lines = [
            "┌─────────────────────────────────────────────────┐",
            "│            Evaluation Metrics                   │",
            "├──────────────────────────┬──────────────────────┤",
            f"│  Accuracy                │  {self.accuracy:.4f}              │",
            f"│  Precision               │  {self.precision:.4f}              │",
            f"│  Recall                  │  {self.recall:.4f}              │",
            f"│  F1 Score                │  {self.f1:.4f}              │",
            f"│  ROC-AUC                 │  {self.roc_auc:.4f}              │",
            f"│  PR-AUC                  │  {self.pr_auc:.4f}              │",
            f"│  Partial AUC (0.1)       │  {self.p_auc:.4f}              │",
            f"│  Log Loss                │  {self.log_loss:.4f}              │",
            f"│  Equal Error Rate (EER)  │  {self.eer:.4f}              │",
            f"│  Best Threshold          │  {self.threshold:.4f}              │",
            "├──────────────────────────┴──────────────────────┤",
            f"│  Confusion Matrix: {self.confusion_matrix}          │",
            "└─────────────────────────────────────────────────┘",
        ]
        return "\n".join(lines)


def compute_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    p_auc_max_fpr: float = 0.1,
) -> EvalMetrics:
    """
    Compute the full evaluation suite.

    Args:
        y_true      : binary ground truth (0/1), shape (N,)
        y_scores    : continuous anomaly probability, shape (N,)
        p_auc_max_fpr: max FPR for partial AUC

    Returns:
        EvalMetrics dataclass
    """
    if len(np.unique(y_true)) < 2:
        raise ValueError("y_true must contain both classes to compute AUC metrics.")

    # Find best threshold via Youden's J statistic
    fpr_arr, tpr_arr, thresholds = roc_curve(y_true, y_scores)
    j_scores = tpr_arr - fpr_arr
    best_thresh = float(thresholds[np.argmax(j_scores)])

    y_pred = (y_scores >= best_thresh).astype(int)
    y_pred_05 = (y_scores >= 0.5).astype(int)

    # EER from DET curve (FPR at operating point where FPR ~= FNR; lower is better)
    fpr_det, fnr_det, _ = det_curve(y_true, y_scores)
    eer_idx = np.nanargmin(np.abs(fnr_det - fpr_det))
    eer = float(fpr_det[eer_idx])

    # Stable log-loss (clip away exact 0/1 from sigmoid)
    y_clip = np.clip(y_scores.astype(np.float64), 1e-7, 1.0 - 1e-7)

    return EvalMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        roc_auc=float(roc_auc_score(y_true, y_scores)),
        pr_auc=float(average_precision_score(y_true, y_scores)),
        p_auc=float(roc_auc_score(y_true, y_scores, max_fpr=p_auc_max_fpr)),
        log_loss=float(log_loss(y_true, y_clip)),
        eer=eer,
        threshold=best_thresh,
        confusion_matrix=confusion_matrix(y_true, y_pred).tolist(),
        accuracy_at_05=float(accuracy_score(y_true, y_pred_05)),
        f1_at_05=float(f1_score(y_true, y_pred_05, zero_division=0)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
    )
