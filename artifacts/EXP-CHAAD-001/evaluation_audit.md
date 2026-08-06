# Evaluation Pipeline Audit Report

**Experiment:** EXP-CHAAD-001  
**Date:** 2026-07-24  
**Status:** EVALUATION EXPORT FIXED - PROMPT 2 PASSED

## Executive Summary

The confirmed validation prediction-export bug has been fixed. The original
corrupted artifact is retained at `validation_predictions.csv`; the corrected
artifact is saved separately as `validation_predictions_corrected.csv`.

The original exporter generated `sample_id` from `batch_idx * len(labels)`.
That logic collided after a short final DataLoader batch and produced 30
duplicated sample IDs affecting 60 rows. The corrected exporter uses the
dataset-supplied normalized manifest `relative_path` as the stable sample ID.

## Checkpoint Verification

**Status:** SUCCESS

- Checkpoint path: `checkpoints/best_model.pt`
- SHA-256: `7d58293ea5138b730811bb261de0b971a8a5e4e52665961b48f2f4a4d85ad7e9`
- Model state loaded successfully
- Model set to evaluation mode
- No retraining, architecture modification, checkpoint overwrite, or test-set
  evaluation was performed

## Prediction Export Validation

**Status:** PASS

| Check | Value |
|-------|-------|
| Expected validation samples | 28,254 |
| Exported prediction rows | 28,254 |
| Unique sample IDs | 28,254 |
| Duplicate sample-ID count | 0 |
| Duplicate row count | 0 |
| Missing-ID count | 0 |
| Invalid-label count | 0 |
| Non-finite-score count | 0 |
| Non-validation rows | 0 |

Class counts:

| Label | Count |
|-------|-------|
| 0.0 | 22,608 |
| 1.0 | 5,646 |

Machine counts:

| Machine ID | Count |
|------------|-------|
| id_00 | 14,292 |
| id_02 | 13,962 |

## Batch-Size Determinism

**Status:** PASS

The corrected export was generated with batch sizes 16 and 32. After sorting by
`sample_id`:

| Check | Result |
|-------|--------|
| Sample IDs identical | true |
| Labels identical | true |
| Scores equal within tolerance | true |
| Score tolerance | 1e-6 |
| Max absolute score difference | 0.0000002384 |
| ROC-AUC batch 16 | 0.6002609445 |
| ROC-AUC batch 32 | 0.6002609445 |
| ROC-AUC difference | 0.0 |

Note: an initial GPU run without deterministic inference settings showed
batch-shape-dependent numeric drift. The audit script now disables TF32 and
sets deterministic inference controls before model execution.

## Original Corrupted Export Metrics

The original artifact remains available for audit history:
`artifacts/EXP-CHAAD-001/validation_predictions.csv`.

| Metric | Value |
|--------|-------|
| Rows | 28,254 |
| Unique sample IDs | 28,224 |
| Duplicate sample-ID count | 30 |
| Duplicate row count | 60 |
| ROC-AUC | 0.5999999937 |
| Negative-score ROC-AUC | 0.4000000063 |

## Corrected Metrics

Metrics below were recomputed from
`artifacts/EXP-CHAAD-001/validation_predictions_corrected.csv`.

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.6002609445 |
| PR-AUC | 0.2578861055 |
| EER | 0.4264914172 |
| Positive-score ROC-AUC | 0.6002609445 |
| Negative-score ROC-AUC | 0.3997390555 |
| Youden threshold | 0.4995152950 |

Metrics at threshold 0.5:

| Metric | Value |
|--------|-------|
| Accuracy | 0.5444538826 |
| Balanced accuracy | 0.5756161663 |
| Precision | 0.2475717979 |
| Recall | 0.6275239107 |
| F1 | 0.3550633863 |
| Confusion matrix | [[11840, 10768], [2103, 3543]] |

Metrics at Youden threshold:

| Metric | Value |
|--------|-------|
| Accuracy | 0.5441353437 |
| Balanced accuracy | 0.5760815437 |
| Precision | 0.2477684798 |
| Recall | 0.6292950762 |
| F1 | 0.3555488842 |
| Confusion matrix | [[11821, 10787], [2093, 3553]] |

## Score Direction

**Status:** CORRECT

Positive-score ROC-AUC (0.6002609445) is greater than negative-score ROC-AUC
(0.3997390555). Higher scores correspond to higher anomaly probability.

## Subgroup Metrics

### By Machine Type

| Machine Type | ROC-AUC | Count |
|--------------|---------|-------|
| fan | 0.4778261433 | 7,368 |
| pump | 0.6541110506 | 6,795 |
| slider | 0.5668446979 | 8,277 |
| valve | 0.5630984836 | 5,814 |

### By Machine ID

| Machine ID | ROC-AUC | Count |
|------------|---------|-------|
| id_00 | 0.6342979841 | 14,292 |
| id_02 | 0.5635492185 | 13,962 |

## Artifacts

- Corrected predictions: `validation_predictions_corrected.csv`
- Export validation report: `prediction_export_validation.json`
- Batch-size determinism report: `prediction_export_determinism.json`
- Corrected independent metrics: `independent_metrics_corrected.json`
- Corrected subgroup metrics: `subgroup_metrics_corrected.json`

## Conclusion

**FINAL CLASSIFICATION:** EVALUATION EXPORT FIXED - PROMPT 2 PASSED

The validation prediction-export bug is fixed and verified. The corrected
validation ROC-AUC is 0.6002609445. EXP-CHAAD-001 remains provisional as a
research result because no held-out test evaluation or downstream baseline,
legacy-comparison refresh, or multi-seed validation has been performed in this
step.
