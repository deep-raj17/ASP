# Architecture Overview

## High-Level Architecture

The repository follows a layered architecture:

1. Data ingestion and preprocessing
2. Model training and checkpointing
3. Calibration of anomaly scores
4. Evaluation on validation data
5. Inference through UI, API, or batch detector

## Components

### Data Layer
- [data/dataset.py](../data/dataset.py): dataset discovery, split assignment, label parsing, feature extraction.
- [utils/audio_utils.py](../utils/audio_utils.py): audio preprocessing, mel extraction, MFCC extraction, augmentation.
- [utils/data_pipeline.py](../utils/data_pipeline.py): optimized data loading pipeline with caching.

### Model Layer
- [models/hybrid_model.py](../models/hybrid_model.py): hybrid model with CNN backbone, temporal module, attention pooling, projector, classifier, and autoencoder branch.

### Training Layer
- [training/loss.py](../training/loss.py): multi-objective loss combining BCE, contrastive, and reconstruction losses.
- [training/trainer.py](../training/trainer.py): training loop, optimizer, scheduler, checkpointing, validation.

### Inference Layer
- [inference/detector.py](../inference/detector.py): detector that computes four anomaly scores and fuses them.
- [inference/production_detector.py](../inference/production_detector.py): optimized batch production detector.
- [production_api.py](../production_api.py): REST API wrapper.
- [app.py](../app.py): Gradio UI.

### Configuration and Artifacts
- [config.py](../config.py): configuration dataclasses.
- [paths.py](../paths.py): canonical artifact paths.

## Execution Flow

### Training
- [train.py](../train.py) loads the configuration, verifies the dataset, builds the model, and runs [training/trainer.py](../training/trainer.py).

### Calibration
- [calibrate.py](../calibrate.py) loads the best checkpoint, builds a normal-data loader, and fits the detector reference statistics.

### Evaluation
- [evaluate.py](../evaluate.py) loads the best checkpoint, runs the validation set through the model, and computes metrics.

### Inference
- [production_api.py](../production_api.py) or [app.py](../app.py) loads the trained model and calibration state and serves predictions.

## Architectural Strengths

- Clear separation between model, scoring, and deployment components.
- Reusable utility layer for audio and validation.
- Production-focused deployment wrappers.

## Architectural Risks

- The architecture is broad but not fully unified around a single experimental protocol.
- Some components appear to overlap in purpose, especially around preprocessing and inference scoring.
- The system currently uses fixed fusion weights rather than a learned or condition-aware mechanism.
