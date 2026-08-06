# Statistical Analysis Plan

## Units and pairing

The primary unit is the training seed. Methods using a shared backbone are
paired by seed and sample. Prediction-level resampling is stratified by machine
type, machine ID, noise condition, and label; it does not replace seed-level
uncertainty.

## Primary analysis

For reliability-aware fusion versus the strongest prespecified fixed
comparator:

1. compute validation ROC-AUC for each of seeds 42, 123, and 2026;
2. compute paired seed-level differences;
3. report mean, standard deviation, median, minimum, maximum, and 95% interval;
4. run the exact paired sign/permutation test supported by three pairs;
5. supplement with 10,000 stratified paired bootstrap replicates over
   prediction rows within each seed.

With only three seeds, p-values have coarse resolution. Conclusions therefore
emphasize paired effect sizes and intervals; lack of significance is not
relabelled as equivalence.

## Secondary analyses

- PR-AUC, standardized pAUC at FPR 0.1, EER, balanced accuracy, and F1.
- Machine type, machine ID, and noise subgroup effects.
- Threshold sensitivity around the validation-selected Youden threshold.
- False-positive and false-negative composition.
- Runtime, parameter count, and peak GPU memory.

## Multiplicity

The primary comparison is singular and prespecified. Secondary pairwise
comparisons use Holm correction within each metric family at alpha 0.05.

## Failure handling

All planned seeds remain in the registry. Infrastructure retries preserve the
same seed and configuration. Scientific failures, NaNs, divergence, and
resource failures are reported; no failed run is silently replaced. Aggregate
tables show both planned and successful run counts.

## Contribution rule

- SUPPORTED: positive effects across all seeds, positive primary interval,
  no material subgroup reversal, and corrected primary evidence supports the
  alternative.
- PARTIALLY SUPPORTED: positive practical effect but uncertainty or subgroup
  inconsistency prevents the stronger verdict.
- NOT SUPPORTED: effect is negligible, inconsistent, or adverse.
- REQUIRES REDEFINITION: mechanism is non-functional or evidence shows the
  claimed contribution is not what drives performance.
