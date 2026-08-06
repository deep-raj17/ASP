# Metrics Registry — CHAAD Project

> Central registry of all evaluation metrics. Every metric MUST have evidence or be marked UNVERIFIED.

---

## Registry

### Accuracy

| Field | Value |
|-------|-------|
| Metric name | Accuracy |
| Definition | `(TP + TN) / (TP + TN + FP + FN)` |
| Implementation | `sklearn.metrics.accuracy_score` |
| Location | `utils/metrics.py` line 199 |
| Input | Binary predictions from thresholded scores |
| Threshold dependence | Yes (Youden's J on validation) |
| Current value | UNVERIFIED |
| CI method | Bootstrap (1000 stratified resamples) |
| Status | UNVERIFIED (no test-set evaluation) |

### Precision

| Field | Value |
|-------|-------|
| Metric name | Precision |
| Definition | `TP / (TP + FP)` |
| Implementation | `sklearn.metrics.precision_score` (zero_division=0) |
| Location | `utils/metrics.py` line 200 |
| Threshold dependence | Yes |
| Current value | UNVERIFIED |
| Status | UNVERIFIED |

### Recall

| Field | Value |
|-------|-------|
| Metric name | Recall |
| Definition | `TP / (TP + FN)` |
| Implementation | `sklearn.metrics.recall_score` (zero_division=0) |
| Location | `utils/metrics.py` line 201 |
| Threshold dependence | Yes |
| Current value | UNVERIFIED |
| Status | UNVERIFIED |

### F1 Score

| Field | Value |
|-------|-------|
| Metric name | F1 Score |
| Definition | `2 * (precision * recall) / (precision + recall)` |
| Implementation | `sklearn.metrics.f1_score` |
| Location | `utils/metrics.py` line 202 |
| Threshold dependence | Yes |
| Current value | UNVERIFIED |
| Status | UNVERIFIED |

### ROC-AUC

| Field | Value |
|-------|-------|
| Metric name | Area Under ROC Curve |
| Definition | Area under the Receiver Operating Characteristic curve (TPR vs FPR) |
| Implementation | `sklearn.metrics.roc_auc_score` |
| Location | `utils/metrics.py` line 204 |
| Input | Continuous anomaly scores (torch.sigmoid(logits)), not binary labels |
| Averaging | N/A (binary) |
| Threshold dependence | No (threshold-agnostic) |
| Current value (legacy) | 0.999999692 (validation only, LEGACY — see ADR-007) |
| Current value (official) | UNVERIFIED (no test-set evaluation) |
| CI method | Bootstrap + DeLong test for method comparison |
| Status | UNVERIFIED for official result |

### PR-AUC

| Field | Value |
|-------|-------|
| Metric name | Area Under Precision-Recall Curve |
| Definition | Average precision across recall levels |
| Implementation | `sklearn.metrics.average_precision_score` |
| Location | `utils/metrics.py` line 205 |
| Input | Continuous anomaly scores |
| Threshold dependence | No |
| Current value (legacy) | 0.999998678 (validation only, LEGACY) |
| Current value (official) | UNVERIFIED |
| CI method | Bootstrap |
| Status | UNVERIFIED |

### Partial AUC (pAUC)

| Field | Value |
|-------|-------|
| Metric name | Partial AUC (max FPR = 0.1) |
| Definition | ROC-AUC restricted to FPR ≤ 0.1 region |
| Implementation | `sklearn.metrics.roc_auc_score(max_fpr=0.1)` |
| Location | `utils/metrics.py` line 206 |
| Threshold dependence | No |
| Current value | UNVERIFIED |
| Status | UNVERIFIED |

### Equal Error Rate (EER)

| Field | Value |
|-------|-------|
| Metric name | Equal Error Rate |
| Definition | Operating point where FPR ≈ FNR; lower is better |
| Implementation | `sklearn.metrics.det_curve` → `argmin(|fnr - fpr|)` |
| Location | `utils/metrics.py` lines 167-169 |
| Threshold dependence | No (operating-point independent) |
| Current value | UNVERIFIED |
| Status | UNVERIFIED |

### Balanced Accuracy

| Field | Value |
|-------|-------|
| Metric name | Balanced Accuracy |
| Definition | `(TPR + TNR) / 2` |
| Implementation | `sklearn.metrics.balanced_accuracy_score` |
| Location | `utils/metrics.py` line 213 |
| Threshold dependence | Yes |
| Current value | UNVERIFIED |
| Status | UNVERIFIED |

### Log Loss

| Field | Value |
|-------|-------|
| Metric name | Log Loss (Binary Cross-Entropy) |
| Definition | `-mean(y*log(p) + (1-y)*log(1-p))` |
| Implementation | `sklearn.metrics.log_loss` (clipped to [1e-7, 1-1e-7]) |
| Location | `utils/metrics.py` line 208 |
| Threshold dependence | No |
| Current value | UNVERIFIED |
| Status | UNVERIFIED |

### Confusion Matrix

| Field | Value |
|-------|-------|
| Metric name | Confusion Matrix |
| Definition | 2×2 matrix: [[TN, FP], [FN, TP]] |
| Implementation | `sklearn.metrics.confusion_matrix` |
| Location | `utils/metrics.py` line 209 |
| Threshold dependence | Yes |
| Current value | UNVERIFIED |
| Status | UNVERIFIED |

### Latency

| Field | Value |
|-------|-------|
| Metric name | Inference latency (single sample) |
| Definition | Wall-clock time for detect() call |
| Implementation | `benchmark.py` `benchmark_single_inference()` |
| Current value | UNVERIFIED |
| Status | UNVERIFIED |

### Parameter Count

| Field | Value |
|-------|-------|
| Metric name | Trainable Parameters |
| Definition | `sum(p.numel() for p in model.parameters() if p.requires_grad)` |
| Implementation | `train.py` line 83 |
| Current value | UNVERIFIED (logged during training) |
| Status | UNVERIFIED |

---

## Legacy Values (from superseded evaluation)

Stored in `checkpoints/eval_report.json` (LEGACY — validation split only, unknown split protocol):

| Metric | Legacy Value | Caveat |
|--------|-------------|--------|
| ROC-AUC | 0.999999692 | Validation set, unknown protocol |
| PR-AUC | 0.999998678 | Validation set, unknown protocol |
| Accuracy | 0.999623919 | Threshold optimized on same split |
| Precision | 0.998010610 | Threshold optimized on same split |
| Recall | 1.0 | Threshold optimized on same split |
| F1 | 0.999004315 | Threshold optimized on same split |
| Threshold | 0.313720703 | Selected on validation |
| Confusion Matrix | [[6469, 3], [0, 1505]] | Validation set |

**These values are NOT valid final results.** They were computed on validation, not test, and the split protocol is unknown. See ADR-007.

---

*Status: All metrics marked UNVERIFIED pending first test-set evaluation under the current machine-independent 3-split protocol. See `docs/DECISIONS.md` ADR-007 for legacy metrics treatment.*
