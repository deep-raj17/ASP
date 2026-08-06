# Frozen Research Question

Freeze date: 2026-07-27

## Primary question

On the machine-independent MIMII validation split, does sample-dependent
reliability-aware fusion of four calibrated anomaly signals improve ROC-AUC
over the strongest prespecified fixed-fusion comparator when both methods use
identical backbone outputs, calibration data, samples, and seeds?

## Principal contribution

A lightweight gate assigns per-sample weights to reconstruction, embedding,
Mahalanobis, and contrastive anomaly signals using the score vector, learned
audio embedding summary, machine type, and noise condition.

## Hypotheses

- Null: the mean paired validation ROC-AUC difference between reliability-aware
  fusion and the strongest fixed comparator is less than or equal to zero.
- Alternative: the mean paired validation ROC-AUC difference is greater than
  zero.

The strongest fixed comparator is selected on validation only from equal
fusion and a prespecified validation-optimized global fusion procedure. This
selection rule is applied identically for every seed and is frozen before new
results.

## Evidence standard

The contribution is not supported merely because one seed improves. Support
requires positive paired effects across the frozen seed set, uncertainty
reporting, acceptable failure accounting, and no materially harmful subgroup
pattern. Statistical significance is reported only if the prespecified test
supports it after correction.
