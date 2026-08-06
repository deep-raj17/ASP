# Result Recalculation Audit

## Eligible prediction artifacts

The only publication-relevant usable export is
`artifacts/EXP-CHAAD-001/validation_predictions_corrected.csv`. It contains
28,254 rows and 28,254 unique normalized sample IDs; labels are binary, scores
are finite, every row is marked `val`, and machine IDs are exactly `id_00` and
`id_02`. Counts match the manifest. No training or protected-test row was
observed.

The original export has 28,254 rows but only 28,224 unique IDs. It is excluded
from scientific reuse. `reports/test_predictions.csv` also has 28,254 unique
rows for `id_00`/`id_02`; it is treated as a provisional validation artifact,
not a test artifact.

## Independently recomputed overall results

| Metric | Corrected validation result |
|---|---:|
| ROC-AUC | 0.6002609444987201 |
| PR-AUC / average precision | 0.25788610546048196 |
| EER, nearest sampled ROC crossing | 0.42651353324563995 |
| Youden threshold | 0.4995152950286865 |
| Accuracy at 0.5 | 0.544453882636087 |
| Balanced accuracy at 0.5 | 0.5756161662654282 |
| Precision at 0.5 | 0.2475717979176857 |
| Recall at 0.5 | 0.6275239107332625 |
| Specificity at 0.5 | 0.5237084217975938 |
| F1 at 0.5 | 0.3550633862805031 |
| Confusion matrix at 0.5 | TN=11840, FP=10768, FN=2103, TP=3543 |

At the validation-selected Youden threshold, balanced accuracy is
0.5760815437 and F1 is 0.3555488842. These results reproduce the stored values.

### EER convention discrepancy

The stored report gives EER `0.4264914171805303`; a nearest-sampled-point
calculation gives `0.42651353324563995`. This small difference is attributable
to the EER crossing/interpolation convention, not different predictions.
Future protocols must freeze the precise EER definition.

## Subgroups

Machine-type ROC-AUC ranges from 0.4778261433 (fan) to 0.6541110506 (pump).
Machine-ID ROC-AUC is 0.6342979841 for `id_00` and 0.5635492185 for `id_02`.
The corrected export does not contain an explicit normalized noise-condition
column beyond `snr`; no new per-noise calculations were added here because the
existing corrected subgroup JSON reports only machine type and machine ID.

## Comparison with reported results

- Corrected stored overall and machine subgroup metrics agree with independent
  recomputation to displayed precision, except the documented EER convention.
- The original historical near-perfect metrics are contradicted by
  identity-correct machine-independent validation evidence.
- The old corrupt export yields ROC-AUC 0.5999999937, close numerically but
  scientifically invalid because row identity is not unique.
- The misleading `reports/test_predictions.csv` yields ROC-AUC 0.6002711760,
  but it is not protected-test evidence.

No protected-test metric was computed or audited.
