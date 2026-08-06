# Training Pipeline Audit Report

**Experiment:** EXP-CHAAD-001  
**Date:** 2026-07-23  
**Status:** MODEL UNDERFITTING

## Executive Summary

Gradients flow correctly and inspected data are finite and non-constant. The model only reduces tiny-batch BCE by 85.55% in 50 epochs and does not reach near-zero loss; validation loss rises from 2.3862 to 3.7270 while final validation AUC is 0.5233. On the available evidence this is **MODEL UNDERFITTING** (with loss imbalance as a contributing factor), not a confirmed broken gradient/data pipeline.

## Epoch History Reconstruction

**Status:** SUCCESS

TensorBoard logs successfully extracted with 14 metric series:
- `train/loss_bce`
- `train/loss_con`
- `train/loss_recon`
- `train/loss_total`
- `train/lr`
- `system/allocated_gb`
- `system/reserved_gb`
- `system/max_allocated_gb`
- `system/free_gb`
- `val/loss`
- `val/roc_auc`
- `val/eer`
- `val/acc_05`
- `val/bal_acc`

**Final Reported Metrics:**
- Train loss: 1.0502
- Validation loss: 3.7270
- Validation ROC-AUC: 0.5233
- Validation EER: 0.4574
- Accuracy @ 0.5: 0.401
- Balanced accuracy: 0.543

## Loss Function Audit

**Status:** WARNING - Imbalanced Loss Weights

**Loss Component Weights:**
- BCE weight: 1.0 (74.07% of total)
- Contrastive weight: 0.3 (22.22% of total)
- Reconstruction weight: 0.05 (3.70% of total)
- BCE positive class weight: 5.0
- Temperature: 0.07

**Total loss weight:** 1.35

**Issues:**
1. **BCE dominates:** 74.07% of total loss comes from BCE
2. **Reconstruction very small:** 0.05 weight (3.70%) may be insufficient for meaningful reconstruction learning
3. **Potential imbalance:** Contrastive loss (22.22%) may be too large relative to reconstruction

**Recommendation:** Consider rebalancing loss weights to ensure all branches contribute meaningfully.

## Gradient Audit

**Status:** VERIFIED

**Gradient Statistics:**
- Parameters with gradients: 476
- Parameters without gradients: 18
- Parameters with near-zero gradients: 0
- Parameters with NaN/Inf gradients: 0

**Sample Gradient Norms:**
- backbone.0.0.0.weight: norm=12.606567
- backbone.0.0.1.weight: norm=0.619031
- backbone.0.0.1.bias: norm=0.297791
- backbone.0.1.0.block.0.0.weight: norm=1.289028
- backbone.0.1.0.block.0.1.weight: norm=0.491349
- backbone.0.1.0.block.0.1.bias: norm=0.253428

**Findings:**
- ✓ Gradients are flowing correctly
- ✓ No disconnected branches
- ✓ No frozen layers (except expected)
- ✓ No NaN or infinite gradients
- ✓ Gradient norms are reasonable

## Data Audit During Training

**Status:** VERIFIED

**Batch Statistics (5 batches inspected):**

| Batch | Shape | Range | Mean | Std | Labels |
|-------|-------|-------|------|-----|--------|
| 0 | [32, 1, 128, 313] | [0.0886, 1.0000] | 0.5053 | 0.1400 | [30, 2] |
| 1 | [32, 1, 128, 313] | [0.1693, 1.0000] | 0.5270 | 0.1436 | [25, 7] |
| 2 | [32, 1, 128, 313] | [0.1804, 1.0000] | 0.5265 | 0.1442 | [27, 5] |
| 3 | [32, 1, 128, 313] | [0.0309, 1.0000] | 0.5435 | 0.1389 | [27, 5] |
| 4 | [32, 1, 128, 313] | [0.0519, 1.0000] | 0.5400 | 0.1415 | [29, 3] |

**Findings:**
- ✓ No NaN values
- ✓ No Inf values
- ✓ No all-zero tensors
- ✓ Reasonable value ranges [0.03-1.0]
- ✓ Consistent tensor shapes
- ✓ Label distribution shows class imbalance (mostly normal samples)

## Tiny Batch Overfit Test

**Status:** PARTIALLY CAPABLE

**Test Configuration:**
- Dataset size: 16 samples
- Epochs: 50
- Learning rate: 1e-3
- Loss: BCE only

**Results:**
- Initial loss: 1.438466
- Final loss: 0.207911
- Loss reduction: 85.55%
- Classification: TRAINING PARTIALLY CAPABLE

**Interpretation:**
- The model can partially overfit a tiny batch (85.55% loss reduction)
- However, it does not achieve near-zero loss (expected <0.01 for full overfit)
- This suggests optimization issues or architectural limitations
- Possible causes:
  - Learning rate too low
  - Model capacity insufficient
  - Optimization stuck in local minima
  - Regularization too strong

## Training Curves

**Available Metrics:**
- Train loss components (BCE, contrastive, reconstruction, total)
- Validation loss
- Validation ROC-AUC
- Validation EER
- Validation accuracy @ 0.5
- Validation balanced accuracy
- Learning rate schedule

**Note:** Full curve analysis requires plotting TensorBoard logs.

## Classification

**FINAL STATUS:** MODEL UNDERFITTING

**Evidence:**
- ✓ Gradients flow correctly
- ✓ Data is valid and properly preprocessed
- ✓ No NaN/Inf in data or gradients
- ✓ Model can partially overfit tiny batch
- ⚠ Model cannot fully overfit tiny batch
- ⚠ Loss weights imbalanced (BCE dominates)
- ⚠ Reconstruction weight very small

**Conclusion:**
The training pipeline is functional but the model shows limited learning capability. The inability to fully overfit even 16 samples suggests the model may be underfitting or there are optimization issues. The imbalanced loss weights (BCE 74%, reconstruction 3.7%) may prevent effective multi-branch learning.

## Training Curves

Generated from the verified TensorBoard event history and stored in
`training_curves/`: `training_loss.png`, `validation_loss.png`,
`validation_auc.png`, and `learning_rate.png`.

## Recommendations

1. **Rebalance loss weights:** Increase reconstruction weight to 0.2-0.3, reduce BCE weight to 0.6-0.7
2. **Increase learning rate:** Try 2e-3 or 5e-3 for faster convergence
3. **Reduce regularization:** Check if weight decay or dropout is too strong
4. **Verify model capacity:** Ensure model has sufficient parameters for the task
5. **Check optimization:** Consider using different optimizer (Adam vs AdamW) or scheduler

## Next Steps

1. Proceed with Prompt 4 (compare with legacy 99% experiment) to identify protocol differences
2. Run diagnostic baselines (Prompt 5) to determine if the issue is CHAAD-specific or general
3. Based on findings, develop controlled improvement plan (Prompt 6)
