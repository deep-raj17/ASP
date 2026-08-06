# Training Pipeline

## Entry Point

- [train.py](../train.py)

The entry point verifies the dataset path, creates output directories, builds the model, resolves resumable checkpoints if present, and launches the trainer.

## Dataset Preparation

- [data/dataset.py](../data/dataset.py)
- [utils/audio_utils.py](../utils/audio_utils.py)

The dataset loader scans the MIMII tree, parses labels from the directory structure, and builds train and validation splits.

## Training Loop

- [training/trainer.py](../training/trainer.py)

The trainer:
- moves the model to the selected device,
- builds an AdamW optimizer,
- creates a scheduler,
- runs the training epoch loop,
- runs validation after each epoch,
- saves checkpoints,
- and tracks the best validation checkpoint.

## Optimization Strategy

The trainer uses:
- mixed precision when available,
- gradient accumulation,
- gradient clipping,
- and a multi-objective loss.

## Checkpointing

The trainer saves:
- epoch checkpoints,
- best-model checkpoints,
- optimizer state,
- scheduler state,
- and scaler state.

## Validation and Metrics

Validation metrics are computed using [utils/metrics.py](../utils/metrics.py) and include ROC-AUC, PR-AUC, accuracy, recall, F1, EER, and log loss.

## Notes

The current training pipeline is functional, but it is not yet framed around a strict publication-grade train/validation/test protocol. The repository documentation itself already notes the need for a separate test set and proper experimental protocol.
