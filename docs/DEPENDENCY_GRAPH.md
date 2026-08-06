# Dependency Graph

## Import Overview

The repository is organized around a relatively straightforward dependency structure:

- [train.py](../train.py) imports configuration, dataset utilities, model code, and trainer.
- [calibrate.py](../calibrate.py) imports configuration, dataset utilities, model code, and the detector.
- [evaluate.py](../evaluate.py) imports configuration, dataset code, model code, and metric utilities.
- [production_api.py](../production_api.py) imports configuration, artifact paths, model code, detector code, audio utilities, validation utilities, and GPU helpers.
- [app.py](../app.py) imports the production detector, audio utilities, and configuration.

## Main Dependency Paths

### Training Path
- [train.py](../train.py)
  - [config.py](../config.py)
  - [data/dataset.py](../data/dataset.py)
  - [models/hybrid_model.py](../models/hybrid_model.py)
  - [training/trainer.py](../training/trainer.py)

### Calibration Path
- [calibrate.py](../calibrate.py)
  - [config.py](../config.py)
  - [data/dataset.py](../data/dataset.py)
  - [models/hybrid_model.py](../models/hybrid_model.py)
  - [inference/detector.py](../inference/detector.py)

### Evaluation Path
- [evaluate.py](../evaluate.py)
  - [config.py](../config.py)
  - [data/dataset.py](../data/dataset.py)
  - [models/hybrid_model.py](../models/hybrid_model.py)
  - [utils/metrics.py](../utils/metrics.py)

### Inference Path
- [production_api.py](../production_api.py)
  - [config.py](../config.py)
  - [paths.py](../paths.py)
  - [models/hybrid_model.py](../models/hybrid_model.py)
  - [inference/production_detector.py](../inference/production_detector.py)
  - [utils/audio_utils.py](../utils/audio_utils.py)
  - [utils/validation.py](../utils/validation.py)

## Observations

- The architecture is mostly top-down and explicit.
- There is a moderate amount of cross-module coupling, mostly around configuration and audio preprocessing.
- The detector logic is reused by both the standard detector and the production detector.

## Risks

- Shared global configuration can make behavior harder to trace across experiments.
- Several components depend heavily on the same configuration object and artifact paths.
- The repository would benefit from more explicit module boundaries between research experimentation and production inference.
