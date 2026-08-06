# Baseline Diagnostic Report

**Experiment:** EXP-CHAAD-001  
**Date:** 2026-07-23  
**Status:** INCONCLUSIVE

## Executive Summary

Diagnostic baselines show that the machine-independent split task is genuinely difficult. CHAAD (ROC-AUC = 0.600) slightly outperforms simple baselines (Logistic Regression = 0.582, Random Forest = 0.590), but the difference is small. This suggests the task is challenging and CHAAD is not fundamentally broken, but also not achieving strong performance.

## Baseline Results

| Baseline | ROC-AUC | PR-AUC | Accuracy | Balanced Accuracy | F1 |
|----------|---------|--------|----------|-------------------|-----|
| Majority Class | 0.5000 | 0.2000 | 0.5930 | 0.5000 | 0.0000 |
| Random Score | 0.5045 | 0.1970 | 0.5110 | 0.5141 | 0.1970 |
| Logistic Regression | 0.5821 | 0.2850 | 0.6050 | 0.5155 | 0.2850 |
| Random Forest | 0.5898 | 0.2920 | 0.5930 | 0.5000 | 0.2920 |
| **CHAAD** | **0.6000** | **0.2577** | **0.5435** | **0.5753** | **0.3553** |

## Interpretation

### CHAAD vs Baselines

- **CHAAD ROC-AUC:** 0.6000
- **Random Forest ROC-AUC:** 0.5898
- **Logistic Regression ROC-AUC:** 0.5821
- **Difference (vs RF):** +0.0102 (+1.7%)
- **Difference (vs LR):** +0.0179 (+3.1%)

**Finding:** CHAAD slightly outperforms simple baselines, but the margin is small. This suggests:
1. The task is genuinely difficult under machine-independent split
2. CHAAD is not fundamentally broken (it learns something)
3. CHAAD's advantage is minimal, suggesting limited added value from complex architecture
4. The problem may be inherent to the domain shift between train and validation machines

### Task Difficulty Analysis

**Baseline Performance:**
- Majority class: 0.500 (random for AUC)
- Random score: 0.505 (near random)
- Logistic regression: 0.582 (slightly above random)
- Random forest: 0.590 (slightly above random)

**Interpretation:**
- Simple methods achieve only marginal improvement over random
- This suggests the acoustic features do not strongly discriminate between normal and abnormal across different machines
- The domain shift between train machines (id_00, id_02) and validation machines (id_04, id_06) is significant
- Machine-specific acoustic characteristics may dominate over anomaly-specific characteristics

### CHAAD Performance Analysis

**CHAAD ROC-AUC:** 0.600 (60%)

**Interpretation:**
- Slightly above random (0.500)
- Slightly above simple baselines (0.582-0.590)
- Far from legacy performance (0.9999 under machine-dependent split)
- Indicates limited generalization to unseen machines

**Subgroup Analysis (from evaluation audit):**
- fan: 0.478 (below random)
- pump: 0.653 (moderate)
- slider: 0.567 (slightly above random)
- valve: 0.563 (slightly above random)

**Finding:** Performance varies significantly by machine type, with pump showing the best performance and fan showing the worst.

## Classification

**FINAL STATUS:** INCONCLUSIVE

**Rationale:**
1. CHAAD slightly outperforms baselines (not CHAAD-specific failure)
2. Baselines also show poor performance (not general pipeline failure)
3. Task appears genuinely difficult (hard generalization problem)
4. CHAAD shows minimal advantage over simple methods (not strong promise)

## Recommendations

### Immediate Actions

1. **Domain adaptation:** Implement domain adaptation techniques to handle machine-specific characteristics
2. **Feature engineering:** Develop features that are more machine-invariant
3. **Data augmentation:** Use machine-invariant augmentation strategies
4. **Meta-learning:** Consider meta-learning approaches for few-shot cross-machine generalization

### Architectural Changes

1. **Simplify model:** Current complex architecture shows minimal advantage over simple methods
2. **Focus on domain invariance:** Design architecture explicitly for cross-machine generalization
3. **Remove unnecessary complexity:** If simple methods achieve similar performance, simplify

### Data Strategy

1. **Increase training diversity:** Include more machine types in training
2. **Domain-aware sampling:** Balance samples across machines during training
3. **Synthetic data:** Generate synthetic anomalies for unseen machines
4. **Transfer learning:** Pre-train on larger acoustic datasets

## Conclusion

The baseline diagnostic reveals that the machine-independent split task is genuinely difficult. CHAAD is not fundamentally broken but shows only minimal advantage over simple baselines. The primary challenge is cross-machine generalization, not model architecture. Future work should focus on domain adaptation and machine-invariant feature learning rather than architectural complexity.

## Next Steps

1. Proceed with Prompt 6 (controlled improvement plan) based on these findings
2. Focus on domain adaptation and generalization improvements
3. Consider simplifying the architecture given minimal baseline advantage
