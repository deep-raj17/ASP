# CHAAD Improvement Plan

**Experiment:** EXP-CHAAD-001  
**Date:** 2026-07-23  
**Based on:** Prompts 1-5 diagnostic findings

## Summary of Diagnostic Findings

### Key Insights

1. **Primary Issue:** Cross-machine generalization failure (machine-independent split)
   - Legacy (machine-dependent): 99.99% ROC-AUC
   - Current (machine-independent): 52-60% ROC-AUC
   - Performance drop: ~47%

2. **Pipeline Status:** VERIFIED CORRECT
   - Evaluation pipeline: No bugs found
   - Training pipeline: Partially verified (can partially overfit)
   - Data: No issues detected
   - Gradients: Flowing correctly

3. **Baseline Comparison:** Task is genuinely difficult
   - CHAAD: 0.600 ROC-AUC
   - Logistic Regression: 0.582 ROC-AUC
   - Random Forest: 0.590 ROC-AUC
   - CHAAD shows minimal advantage (1.7-3.1%)

4. **Training Issues:**
   - Loss weights imbalanced (BCE 74%, reconstruction 3.7%)
   - Model cannot fully overfit tiny batch (85.55% reduction, not >99%)
   - Possible optimization issues

## Ranked Suspected Causes

### 1. Domain Shift Across Machine IDs (PRIMARY CAUSE)
- **Evidence:** 47% performance drop when changing from machine-dependent to machine-independent split
- **Evidence:** Baselines also perform poorly (0.58-0.59)
- **Evidence:** Subgroup performance varies significantly (fan: 0.48, pump: 0.65)
- **Hypothesis:** Acoustic characteristics are machine-specific, not anomaly-specific

### 2. Loss Weight Imbalance (SECONDARY CAUSE)
- **Evidence:** BCE dominates 74% of total loss
- **Evidence:** Reconstruction weight very small (3.7%)
- **Hypothesis:** Multi-branch learning not effective due to imbalance

### 3. Insufficient Domain Adaptation (TERTIARY CAUSE)
- **Evidence:** Minimal advantage over simple baselines
- **Evidence:** Complex architecture not helping generalization
- **Hypothesis:** Model not designed for cross-machine generalization

### 4. Optimization Issues (QUATERNARY CAUSE)
- **Evidence:** Cannot fully overfit tiny batch (85.55% vs expected >99%)
- **Evidence:** Learning rate may be too low
- **Hypothesis:** Optimization stuck in local minima

## Proposed Experiments

### Experiment Matrix (First Cycle)

| ID | Hypothesis | Configuration | Expected Benefit | Risk | Duration |
|----|------------|---------------|------------------|------|----------|
| EXP-IMP-001 | Loss weight rebalancing will improve multi-branch learning | BCE=0.6, Contrastive=0.2, Recon=0.2 | +5-10% ROC-AUC | Low | 2-3 hours |
| EXP-IMP-002 | Domain-aware sampling will improve cross-machine generalization | Balance samples by machine ID during training | +10-15% ROC-AUC | Low | 2-3 hours |
| EXP-IMP-003 | Higher learning rate will improve optimization | LR=5e-3 (vs 1e-4) | +3-5% ROC-AUC | Medium | 2-3 hours |
| EXP-IMP-004 | Classification-only will match current performance | Remove reconstruction and contrastive branches | 0% change (baseline) | Low | 1-2 hours |
| EXP-IMP-005 | Machine-invariant augmentation will improve generalization | Add SpecAugment with machine-aware parameters | +5-10% ROC-AUC | Medium | 2-3 hours |

## Detailed Experiment Plans

### EXP-IMP-001: Loss Weight Rebalancing

**Hypothesis:** Rebalancing loss weights will enable effective multi-branch learning and improve anomaly detection.

**Evidence:**
- Current: BCE=1.0 (74%), Contrastive=0.3 (22%), Recon=0.05 (3.7%)
- Reconstruction too small to contribute meaningfully
- Imbalanced weights may prevent effective fusion

**Configuration:**
```python
bce_weight = 0.6          # Reduced from 1.0
contrastive_weight = 0.2  # Reduced from 0.3
recon_weight = 0.2        # Increased from 0.05
```

**Expected Benefit:** +5-10% ROC-AUC (0.60 → 0.63-0.66)

**Scientific Risk:** Low - only changes loss weights, not architecture

**Validation Criterion:** ROC-AUC > 0.65 on validation set

**Rollback Rule:** If ROC-AUC < 0.58, revert to original weights

**Code Area:** `config.py` lines 99-102

---

### EXP-IMP-002: Domain-Aware Sampling

**Hypothesis:** Balancing samples by machine ID during training will improve cross-machine generalization.

**Evidence:**
- Current: Natural sampling (imbalanced by machine ID)
- Subgroup performance varies (fan: 0.48, pump: 0.65)
- Domain shift is primary issue

**Configuration:**
- Modify DataLoader to sample equally from each machine ID
- Ensure each batch contains samples from all training machines
- Weighted sampler by machine ID

**Expected Benefit:** +10-15% ROC-AUC (0.60 → 0.66-0.69)

**Scientific Risk:** Low - only changes sampling, not model

**Validation Criterion:** ROC-AUC > 0.65 on validation set

**Rollback Rule:** If ROC-AUC < 0.58, revert to natural sampling

**Code Area:** `data/dataset.py` (add WeightedRandomSampler)

---

### EXP-IMP-003: Higher Learning Rate

**Hypothesis:** Increasing learning rate will improve optimization and enable better convergence.

**Evidence:**
- Current: LR=1e-4
- Model cannot fully overfit tiny batch (85.55% vs >99% expected)
- May be stuck in local minima

**Configuration:**
```python
learning_rate = 5e-3  # Increased from 1e-4
```

**Expected Benefit:** +3-5% ROC-AUC (0.60 → 0.62-0.63)

**Scientific Risk:** Medium - higher LR may cause instability

**Validation Criterion:** ROC-AUC > 0.62 on validation set

**Rollback Rule:** If training loss diverges or ROC-AUC < 0.55, revert to 1e-4

**Code Area:** `config.py` line 91

---

### EXP-IMP-004: Classification-Only Baseline

**Hypothesis:** Classification-only (without reconstruction and contrastive) will match current performance, indicating multi-branch adds no value.

**Evidence:**
- CHAAD shows minimal advantage over simple baselines (1.7-3.1%)
- Complex architecture may be unnecessary
- Reconstruction weight very small (3.7%)

**Configuration:**
```python
bce_weight = 1.0
contrastive_weight = 0.0  # Disabled
recon_weight = 0.0        # Disabled
```

**Expected Benefit:** 0% change (establishes baseline)

**Scientific Risk:** Low - simplifies architecture

**Validation Criterion:** ROC-AUC within ±0.02 of current (0.58-0.62)

**Rollback Rule:** N/A (this is a baseline experiment)

**Code Area:** `config.py` lines 99-102

---

### EXP-IMP-005: Machine-Invariant Augmentation

**Hypothesis:** Machine-invariant augmentation will improve cross-machine generalization.

**Evidence:**
- Domain shift is primary issue
- Current augmentation may not address machine-specific characteristics
- Need augmentation that preserves anomaly while varying machine characteristics

**Configuration:**
- Add frequency masking with machine-aware parameters
- Add time masking with machine-aware parameters
- Add pitch shifting with small range (±1 semitone)
- Add noise injection with machine-specific SNR

**Expected Benefit:** +5-10% ROC-AUC (0.60 → 0.63-0.66)

**Scientific Risk:** Medium - may remove anomaly-specific information

**Validation Criterion:** ROC-AUC > 0.65 on validation set

**Rollback Rule:** If ROC-AUC < 0.58, revert to original augmentation

**Code Area:** `utils/audio_utils.py` (augmentation functions)

## Recommended Next Experiment

**IMMEDIATE NEXT RUN: EXP-IMP-002 (Domain-Aware Sampling)**

**Justification:**
1. **Highest Information Value:** Addresses the primary cause (domain shift)
2. **Low Risk:** Only changes sampling, not model architecture
3. **High Expected Benefit:** +10-15% ROC-AUC
4. **Fast to Implement:** Simple WeightedRandomSampler addition
5. **Scientifically Sound:** Addresses the core generalization problem

**Expected Duration:** 2-3 hours

**Success Criteria:** ROC-AUC > 0.65 on validation set

**Failure Criteria:** ROC-AUC < 0.58 on validation set

## Stop Conditions

**Stop Improvement Cycle If:**
1. ROC-AUC > 0.75 on validation set (sufficient for publication)
2. 3 consecutive experiments show no improvement (< +2% ROC-AUC)
3. All 5 experiments completed without reaching ROC-AUC > 0.70

**Proceed to Multi-Seed If:**
1. ROC-AUC > 0.70 on validation set
2. Stable configuration identified (consistent across 2+ experiments)

## Long-Term Considerations

If first cycle does not achieve ROC-AUC > 0.70:

1. **Domain Adaptation:** Implement explicit domain adaptation (DANN, ADDA)
2. **Meta-Learning:** Implement MAML or Reptile for few-shot cross-machine learning
3. **Architecture Simplification:** Remove complex branches if they add no value
4. **Feature Engineering:** Develop machine-invariant acoustic features
5. **Data Collection:** Collect more diverse training data

## Acceptance Criteria for Publication

**Minimum Requirements:**
- ROC-AUC > 0.70 on validation set (machine-independent)
- Multi-seed evaluation (5 seeds) with mean ± std reported
- Confidence intervals (bootstrap 95%)
- Comparison with baselines
- Ablation study

**Ideal Requirements:**
- ROC-AUC > 0.80 on validation set
- Test set evaluation (untouched)
- Domain adaptation techniques
- Cross-dataset generalization

## Timeline

**First Improvement Cycle:** 1-2 days (5 experiments)
**Second Cycle (if needed):** 2-3 days (domain adaptation)
**Multi-Seed Validation:** 1-2 days
**Total:** 4-7 days to publication-ready results
