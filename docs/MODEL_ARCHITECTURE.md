# Model Architecture

## Overview

The repository implements a hybrid anomaly detection model centered on a convolutional backbone, a temporal module, a contrastive embedding branch, a classifier head, and a reconstruction branch.

## Main Model

- [models/hybrid_model.py](../models/hybrid_model.py)

### Components

1. CNN Backbone
   - EfficientNet-B0, EfficientNet-B2, EfficientNet-B4, or ResNet-50 can be selected.
   - The first convolution layer is patched to accept one input channel.

2. Temporal Module
   - Either a Transformer encoder or a BiLSTM module.

3. Attention Pooling
   - Learns a weighted pooling over temporal features.

4. Projection Head
   - Produces L2-normalized embeddings used for contrastive learning.

5. Classifier Head
   - Produces a binary anomaly logit.

6. Autoencoder Branch
   - Reconstructs the input spectrogram and provides a reconstruction-based anomaly signal.

## Training Objective

- [training/loss.py](../training/loss.py)

The loss combines:
- BCE with logits for supervised anomaly classification,
- Supervised contrastive loss for embedding learning,
- MSE reconstruction loss for the autoencoder branch.

## Detection Signals

The detector uses four anomaly signals:
- reconstruction error,
- embedding distance,
- Mahalanobis distance,
- contrastive nearest-neighbor distance.

These are computed in [inference/detector.py](../inference/detector.py).

## Assessment

The architecture is multi-branch and expressive, but the novelty of the system should not be claimed simply because several methods are combined. The main scientific question is not the existence of the branches, but whether the fusion and calibration mechanism is genuinely new and validated.
