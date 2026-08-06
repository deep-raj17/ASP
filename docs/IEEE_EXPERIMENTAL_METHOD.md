# IEEE Experimental Method Section

## Dataset

We evaluate on the MIMII (Malfunctioning Industrial Machine Investigation and Inspection) dataset [1], which contains acoustic recordings of industrial machines (fan, pump, slider, valve) under varying noise conditions (-6 dB, 0 dB, 6 dB). The dataset consists of 53,046 audio files, with 43,146 normal samples and 9,900 abnormal samples. Each recording is 10 seconds long at a sampling rate of 16 kHz.

**Note:** This section describes the current experimental setup. For publication, a proper train/validation/test split must be implemented.

## Data Splitting

We use a deterministic hash-based split with seed 42, assigning 15% of files to validation and 85% to training. The split is applied at the file level before any processing. The protocol is machine-dependent, meaning the same machine ID may appear in both train and validation splits. The current implementation does not include a separate test set.

**Critical Issue:** No test set exists. All reported metrics are validation metrics. For publication, a proper train/validation/test split must be implemented with machine-independent protocol.

## Feature Extraction

Audio signals are converted to log-mel spectrograms with the following parameters:
- FFT size: 2048
- Hop length: 512
- Number of mel bands: 128
- Frequency range: 20 Hz to 8 kHz

The log-mel spectrograms are normalized to [0, 1] using fixed constants (offset +80.0, scale /80.0). No global statistics are fitted on the training data. We also extract 40 MFCC coefficients using the same spectrogram parameters.

## Data Augmentation

During training, we apply the following augmentations:
- Additive Gaussian noise (std=0.005)
- SpecAugment: time masking (param=30) and frequency masking (param=15)
- Mixup (α=0.4) with probability 0.3

Augmentations are applied only to the training split.

## Model Architecture

Our model is a hybrid architecture combining:
1. **Backbone:** EfficientNet-B4 pre-trained on ImageNet
2. **Temporal Modeling:** Transformer encoder (4 layers, 8 heads, d_model=256)
3. **Autoencoder Branch:** Reconstruction decoder with latent dimension 128
4. **Embedding Branch:** 256-dimensional embedding with attention pooling

## Training

We train using binary cross-entropy loss with positive class weight 5.0 to handle class imbalance. Additional losses include contrastive loss (weight=0.3) and reconstruction loss (weight=0.05). Training uses:
- Batch size: 32
- Optimizer: AdamW (lr=1e-4, weight_decay=1e-4)
- Scheduler: OneCycleLR
- Mixed precision: FP16
- Gradient accumulation: 2 steps
- Epochs: 100

## Calibration

After training, we fit reference distributions on normal training samples only (37,685 samples). We compute:
- Reconstruction error statistics (μ, σ)
- Embedding cosine distance statistics (μ, σ)
- Mahalanobis distance statistics (μ, σ) with Ledoit-Wolf covariance
- Contrastive k-NN distance statistics (μ, σ)

## Anomaly Scoring

For each sample, we compute four anomaly signals:
1. Reconstruction error (MSE between input and reconstructed spectrogram)
2. Embedding distance (cosine distance to reference mean)
3. Mahalanobis distance (via inverse covariance matrix)
4. Contrastive distance (1 - mean top-k cosine similarity to reference pool)

Each signal is z-score normalized using calibration statistics, mapped to [0,1] via sigmoid, and fused using fixed weights (recon=0.30, embed=0.25, mahal=0.30, contra=0.15).

## Threshold Selection

The decision threshold is selected on the validation set using Youden's J statistic (maximizing sensitivity + specificity - 1). The optimal threshold is 0.3137.

**Critical Issue:** Threshold selection and evaluation both use the validation set. For publication, threshold must be selected on validation and evaluated on a separate test set.

## Evaluation Metrics

We evaluate using:
- ROC-AUC (Area under ROC curve)
- PR-AUC (Area under Precision-Recall curve)
- Partial AUC (max FPR=0.1)
- Accuracy, Precision, Recall, F1
- Balanced Accuracy
- Equal Error Rate (EER)
- Log Loss

Ranking metrics (ROC-AUC, PR-AUC) use continuous anomaly scores. Classification metrics use binary predictions at the optimal threshold.

## Current Results (Validation Set)

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.9999 |
| PR-AUC | 0.9999 |
| Accuracy | 0.9996 |
| Precision | 0.9980 |
| Recall | 1.0000 |
| F1 | 0.9990 |
| Balanced Accuracy | 0.9998 |
| EER | 0.0005 |

**Critical Issue:** These are validation metrics, not test metrics. No test set exists for final evaluation.

## Implementation Details

The implementation uses PyTorch 2.5.1 with CUDA 12.1 on an NVIDIA RTX 4070 SUPER GPU. Training takes approximately X hours on this hardware.

## Limitations

1. **No test set:** All metrics are validation metrics
2. **Threshold selection bias:** Same split used for selection and evaluation
3. **Machine ID overlap:** Same machine IDs appear in train and validation
4. **No confidence intervals:** Statistical uncertainty not quantified
5. **Single seed:** Results from single random seed only
6. **No baselines:** No comparison with simpler methods
7. **No ablation study:** Component contributions not evaluated
8. **No unseen-condition tests:** Generalization not demonstrated

## Required Actions for Publication

1. Create proper train/validation/test split with machine-independent protocol
2. Retrain model with frozen experimental protocol
3. Select threshold on validation, evaluate on test
4. Add bootstrap confidence intervals (2,000+ replicates)
5. Run multi-seed evaluation (5 seeds)
6. Implement baseline comparisons
7. Conduct ablation study
8. Evaluate on unseen machine IDs or noise conditions
9. Perform error analysis
10. Add statistical significance testing

## References

[1] MIMII Dataset: Malfunctioning Industrial Machine Investigation and Inspection
