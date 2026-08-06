# Validation Result Recomputation

## Input

`artifacts/EXP-CHAAD-001/validation_predictions_corrected.csv`

The file contains 28,254 rows, 28,254 unique sample IDs, split value `val`,
machine IDs `id_00` and `id_02`, 22,608 negative labels, 5,646 positive labels,
no duplicate IDs, and finite continuous scores.

## Independently recomputed results

| Metric | Value |
|---|---:|
| ROC-AUC | 0.6002609444987201 |
| PR-AUC | 0.25788610546048196 |
| EER | 0.42648619957537154 |
| Youden threshold | 0.4995152950286865 |
| Accuracy at 0.5 | 0.544453882636087 |
| Balanced accuracy at 0.5 | 0.5756161662654282 |
| Precision at 0.5 | 0.2475717979176857 |
| Recall at 0.5 | 0.6275239107332625 |
| F1 at 0.5 | 0.3550633862805031 |

Confusion matrix at 0.5: `[[11840, 10768], [2103, 3543]]`.

At the Youden threshold, accuracy is 0.5441353436681532, balanced accuracy is
0.5760815437417693, and F1 is 0.35554888421895325.

The ROC-AUC, PR-AUC, threshold, and threshold metrics exactly reproduce
`artifacts/EXP-CHAAD-001/independent_metrics_corrected.json`. The tiny EER
difference from that JSON is attributable to the earlier EER operating-point
implementation; it does not change the underfitting conclusion.

## Scientific interpretation

This is validation-only evidence from one seed and one provisional checkpoint.
It supports neither superiority nor the reliability-aware contribution. The
score is non-degenerate but weak and consistent with underfitting.

No protected test prediction file or test sample was read.
