# Metric Definitions

## Overview

This document defines all metrics used in the acoustic anomaly detection project and their calculation methods.

## Metrics

### ROC-AUC (Receiver Operating Characteristic - Area Under Curve)

**Definition:** Area under the ROC curve, which plots True Positive Rate (TPR) vs False Positive Rate (FPR) at various threshold settings.

**Range:** [0, 1]
- 0.5: Random classifier
- 1.0: Perfect classifier

**Implementation:** `utils/metrics.py` line 118
```python
roc_auc = float(roc_auc_score(y_true, y_scores))
```

**Input:** Continuous anomaly scores (y_scores), not binary labels

**Verification:** ✓ Correct - uses continuous scores

### PR-AUC (Precision-Recall Area Under Curve)

**Definition:** Area under the Precision-Recall curve, which plots Precision vs Recall at various threshold settings.

**Range:** [0, 1]
- Higher is better
- More informative than ROC-AUC for imbalanced datasets

**Implementation:** `utils/metrics.py` line 119
```python
pr_auc = float(average_precision_score(y_true, y_scores))
```

**Input:** Continuous anomaly scores (y_scores), not binary labels

**Verification:** ✓ Correct - uses continuous scores

### Partial AUC (pAUC)

**Definition:** Area under the ROC curve restricted to a maximum false positive rate (max_fpr).

**Parameters:** max_fpr = 0.1 (focuses on low FPR region)

**Implementation:** `utils/metrics.py` line 120
```python
p_auc = float(roc_auc_score(y_true, y_scores, max_fpr=p_auc_max_fpr))
```

**Rationale:** More relevant for applications where false positives must be kept low.

### Accuracy

**Definition:** (TP + TN) / (TP + TN + FP + FN)

**Range:** [0, 1]

**Implementation:** `utils/metrics.py` line 114
```python
accuracy = float(accuracy_score(y_true, y_pred))
```

**Note:** Computed using binary predictions (y_pred) at the optimal threshold (Youden's J).

### Precision

**Definition:** TP / (TP + FP)

**Range:** [0, 1]
- Also called Positive Predictive Value

**Implementation:** `utils/metrics.py` line 115
```python
precision = float(precision_score(y_true, y_pred, zero_division=0))
```

### Recall (Sensitivity)

**Definition:** TP / (TP + FN)

**Range:** [0, 1]
- Also called True Positive Rate or Sensitivity

**Implementation:** `utils/metrics.py` line 116
```python
recall = float(recall_score(y_true, y_pred, zero_division=0))
```

### F1 Score

**Definition:** 2 × (Precision × Recall) / (Precision + Recall)

**Range:** [0, 1]
- Harmonic mean of precision and recall

**Implementation:** `utils/metrics.py` line 117
```python
f1 = float(f1_score(y_true, y_pred, zero_division=0))
```

### Balanced Accuracy

**Definition:** (Sensitivity + Specificity) / 2

**Range:** [0, 1]
- Average of recall for each class
- More robust to class imbalance

**Implementation:** `utils/metrics.py` line 127
```python
balanced_accuracy = float(balanced_accuracy_score(y_true, y_pred))
```

### Log Loss

**Definition:** - (1/N) × Σ[y × log(p) + (1-y) × log(1-p)]

**Range:** [0, ∞)
- 0: Perfect predictions
- Higher: Worse predictions

**Implementation:** `utils/metrics.py` line 121
```python
y_clip = np.clip(y_scores.astype(np.float64), 1e-7, 1.0 - 1e-7)
log_loss = float(log_loss(y_true, y_clip))
```

**Note:** Scores clipped to avoid log(0) errors.

### EER (Equal Error Rate)

**Definition:** False Positive Rate at the operating point where FPR ≈ FNR

**Range:** [0, 1]
- Lower is better
- 0: No errors
- 1: All errors

**Implementation:** `utils/metrics.py` lines 106-108
```python
fpr_det, fnr_det, _ = det_curve(y_true, y_scores)
eer_idx = np.nanargmin(np.abs(fnr_det - fpr_det))
eer = float(fpr_det[eer_idx])
```

### Threshold Selection

**Method:** Youden's J Statistic

**Definition:** J = Sensitivity + Specificity - 1 = TPR - FPR

**Optimal Threshold:** Threshold that maximizes J

**Implementation:** `utils/metrics.py` lines 97-100
```python
fpr_arr, tpr_arr, thresholds = roc_curve(y_true, y_scores)
j_scores = tpr_arr - fpr_arr
best_thresh = float(thresholds[np.argmax(j_scores)])
```

**Current Threshold:** 0.313720703125 (from validation set)

### Confusion Matrix

**Structure:**
```
                Predicted
              Normal  Abnormal
Actual Normal   TN      FP
       Abnormal FN      TP
```

**Implementation:** `utils/metrics.py` line 124
```python
confusion_matrix = confusion_matrix(y_true, y_pred).tolist()
```

**Current Confusion Matrix (Validation):**
```
[[6469, 3],
 [0, 1505]]
```

- True Negatives: 6,469
- False Positives: 3
- False Negatives: 0
- True Positives: 1,505

### Accuracy @ 0.5

**Definition:** Accuracy using fixed threshold of 0.5

**Rationale:** Represents deployment default threshold without optimization

**Implementation:** `utils/metrics.py` line 125
```python
accuracy_at_05 = float(accuracy_score(y_true, y_pred_05))
```

### F1 @ 0.5

**Definition:** F1 score using fixed threshold of 0.5

**Implementation:** `utils/metrics.py` line 126
```python
f1_at_05 = float(f1_score(y_true, y_pred_05, zero_division=0))
```

## Metric Calculation Order

1. Compute continuous anomaly scores from model
2. Compute ROC curve and select optimal threshold (Youden's J)
3. Apply optimal threshold to get binary predictions
4. Compute classification metrics (accuracy, precision, recall, F1)
5. Compute ranking metrics (ROC-AUC, PR-AUC, pAUC) using continuous scores
6. Compute EER from DET curve
7. Compute log loss with clipped scores

## Verification

All metrics have been verified to use continuous anomaly scores for ranking metrics (ROC-AUC, PR-AUC) and binary predictions for classification metrics.

**Code Reference:** `utils/metrics.py` lines 78-128

## Reported Values (Validation Set)

| Metric | Value | Percentage |
|--------|-------|------------|
| ROC-AUC | 0.9999996920 | 99.99997% |
| PR-AUC | 0.9999986781 | 99.99986% |
| Partial AUC (0.1) | 0.9999983790 | 99.99984% |
| Accuracy | 0.9996239188 | 99.96% |
| Precision | 0.9980106101 | 99.80% |
| Recall | 1.0000000000 | 100.00% |
| F1 | 0.9990043146 | 99.90% |
| Balanced Accuracy | 0.9997682324 | 99.98% |
| Log Loss | 0.0048458783 | - |
| EER | 0.0004635352 | 0.046% |

**Note:** These are validation metrics, not test metrics (no test set exists).
