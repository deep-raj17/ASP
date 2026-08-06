# Limitations

## Critical Limitations

### 1. No Separate Test Set

**Issue:** The project has NO separate test set. All reported metrics are validation metrics.

**Impact:**
- Cannot claim generalization to unseen data
- Threshold selection and evaluation both use validation set
- Potential optimistic bias in reported metrics
- Not suitable for publication without test set evaluation

**Required Action:** Create proper train/validation/test split with untouched test set.

### 2. Threshold Selection Bias

**Issue:** Threshold is selected on the same validation set used for evaluation.

**Impact:**
- Optimistic bias in reported metrics
- Does not represent true deployment scenario
- Overestimates model performance

**Required Action:** Select threshold on validation set, freeze it, then evaluate on separate test set.

### 3. Machine ID Overlap

**Issue:** Machine IDs appear in both train and validation splits (machine_dependent protocol).

**Impact:**
- May not represent true generalization to unseen machines
- Model may learn machine-specific features
- Overestimates real-world performance

**Current Overlap:**
- id_00: train (12,191) + val (2,101)
- id_02: train (11,904) + val (2,058)
- id_04: train (10,299) + val (1,746)
- id_06: train (10,828) + val (1,919)

**Required Action:** Implement machine-independent protocol or leave-one-machine-ID-out evaluation.

### 4. No Confidence Intervals

**Issue:** No statistical uncertainty quantification via bootstrap confidence intervals.

**Impact:**
- Cannot assess reliability of point estimates
- No measure of metric stability
- Difficult to compare with other methods

**Required Action:** Implement bootstrap confidence intervals (2,000+ replicates) with proper grouping.

### 5. Single Random Seed

**Issue:** Results from single random seed (split_seed=42). No multi-seed evaluation.

**Impact:**
- Cannot assess result stability across different data splits
- May be lucky/unlucky split
- No measure of variance

**Required Action:** Run experiments with multiple seeds (42, 123, 2026, 3407, 9999) and report mean ± std.

## Methodological Limitations

### 6. No Baseline Comparisons

**Issue:** No comparison with simpler baseline methods.

**Impact:**
- Cannot assess added value of complex architecture
- May be over-engineered for the problem
- Difficult to justify complexity

**Required Baselines:**
- Convolutional autoencoder with reconstruction score
- Basic CNN classifier
- EfficientNet-B4 classifier only
- Reconstruction-only anomaly scoring
- Embedding-distance-only scoring

### 7. No Ablation Study

**Issue:** No evaluation of individual component contributions.

**Impact:**
- Cannot determine which components are essential
- May have unnecessary complexity
- Difficult to interpret model behavior

**Required Ablations:**
- Without reconstruction branch
- Without embedding distance
- Without Mahalanobis score
- Without contrastive nearest-neighbor score
- Without attention pooling
- Without Transformer/BiLSTM
- Equal fusion weights vs learned weights

### 8. No Unseen-Condition Tests

**Issue:** No evaluation on truly unseen machine IDs or noise conditions.

**Impact:**
- Cannot assess true generalization capability
- May overfit to training conditions
- Limited real-world applicability

**Required Tests:**
- Leave-one-machine-ID-out evaluation
- Unseen noise condition evaluation
- Cross-machine-type evaluation

### 9. Supervised vs Unsupervised Ambiguity

**Issue:** Task is supervised binary classification but may be compared with unsupervised methods.

**Impact:**
- Invalid comparisons with published unsupervised methods
- Misleading performance claims
- Ethical concerns in publication

**Required Action:** Clearly disclose supervised setting in any publication.

## Technical Limitations

### 10. Fixed Audio Duration

**Issue:** All audio padded/trimmed to exactly 10 seconds.

**Impact:**
- May lose information from longer recordings
- May introduce artifacts from padding
- Not representative of variable-length real-world audio

### 11. Per-File Processing

**Issue:** Each file treated as single sample (no sliding window segmentation).

**Impact:**
- May miss localized anomalies
- Cannot detect temporal patterns within files
- Limited temporal resolution

### 12. Class Imbalance

**Issue:** Dataset is imbalanced (81.3% normal, 18.7% abnormal).

**Impact:**
- Metrics may be dominated by majority class
- May need different evaluation strategies
- Precision-recall more informative than ROC-AUC

**Current Handling:** BCE with positive class weight (5.0)

### 13. No Error Analysis

**Issue:** No analysis of false positives and false negatives.

**Impact:**
- Cannot understand failure modes
- Cannot identify systematic errors
- Difficult to improve model

**Required Action:** Analyze error cases by machine type, ID, noise condition, and audio characteristics.

## Dataset Limitations

### 14. MIMII Dataset Specifics

**Issue:** Results are specific to MIMII dataset and may not generalize.

**Impact:**
- Limited external validity
- Dataset-specific artifacts may be learned
- Difficult to compare across different industrial settings

### 15. Synthetic Anomalies

**Issue:** MIMII anomalies are artificially introduced, not naturally occurring.

**Impact:**
- May not represent real-world fault patterns
- Anomaly characteristics may be unrealistic
- Limited ecological validity

## Computational Limitations

### 16. Single GPU Evaluation

**Issue:** Results from single GPU (RTX 4070 SUPER).

**Impact:**
- May not scale to other hardware
- Batch size optimization may be hardware-specific
- Reproducibility concerns across different setups

### 17. Windows-Specific Issues

**Issue:** Some optimizations (torch.compile) disabled on Windows.

**Impact:**
- Suboptimal performance
- Different behavior across platforms
- Potential reproducibility issues

## Reporting Limitations

### 18. Excessive Precision

**Issue:** Metrics reported with excessive decimal places (e.g., 99.99997%).

**Impact:**
- Misleading precision
- Not statistically justified
- Unscientific presentation

**Required Action:** Report sensible precision (e.g., ROC-AUC = 0.9999) with confidence intervals.

### 19. No Statistical Significance Testing

**Issue:** No statistical tests to compare with other methods.

**Impact:**
- Cannot claim superiority
- Differences may be due to chance
- Weak scientific claims

**Required Action:** Implement paired statistical tests (DeLong, McNemar, bootstrap).

## Summary

The current results are **NOT suitable for IEEE publication** due to:

1. **No test set** - Critical requirement for publication
2. **Threshold selection bias** - Validation set used for selection and evaluation
3. **No confidence intervals** - Statistical uncertainty not quantified
4. **No multi-seed evaluation** - Result stability not assessed
5. **No baselines** - Cannot justify complexity
6. **No ablation study** - Component contributions unknown
7. **No unseen-condition tests** - Generalization not demonstrated

## Required Actions Before Publication

1. Create proper train/validation/test split
2. Retrain model with frozen protocol
3. Select threshold on validation, evaluate on test
4. Add bootstrap confidence intervals
5. Run multi-seed evaluation
6. Implement baseline comparisons
7. Conduct ablation study
8. Evaluate on unseen conditions
9. Perform error analysis
10. Use sensible precision in reporting
11. Add statistical significance testing
12. Clearly disclose supervised setting

Only after these actions are completed can the results be considered for publication.
