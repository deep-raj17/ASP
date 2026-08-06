# Repository Audit

## Scope

This audit documents the current repository as implemented in the workspace. It is based on the visible source tree, configuration files, training/inference scripts, and existing documentation.

## Repository Overview

The repository is a production-oriented industrial acoustic anomaly detection system built around a hybrid neural network and a multi-score detector. The code is organized around:

- training a hybrid anomaly model,
- calibrating anomaly scores using normal-only reference statistics,
- evaluating on a validation split,
- and serving inference via a Gradio UI, a REST API, or a production detector.

## Top-Level Structure

- app.py: Gradio-based UI entry point.
- production_api.py: Flask REST API for inference.
- train.py: training entry point.
- calibrate.py: calibration entry point.
- evaluate.py: evaluation entry point.
- config.py: global configuration dataclasses.
- paths.py: canonical artifact paths.
- data/: dataset discovery and loading.
- models/: neural network architecture.
- inference/: anomaly detector and production inference engine.
- training/: training objective and trainer.
- utils/: audio processing, validation, metrics, GPU helpers.
- scripts/: auditing, evaluation, and artifact-generation utilities.
- tests/: integrity tests.
- docs/: research and reproducibility notes.

## Important Observations

- The repository is not purely experimental; it includes inference-facing deployment paths and artifact-management logic.
- The project clearly distinguishes training, calibration, evaluation, and production inference.
- The repository already contains audit and reproducibility documentation under docs/.
- Some components are production-oriented, while others are research-oriented; this creates a mixed engineering/research structure.

## Key Modules

### Training
- [train.py](../train.py)
- [training/trainer.py](../training/trainer.py)
- [training/loss.py](../training/loss.py)

### Data
- [data/dataset.py](../data/dataset.py)
- [utils/audio_utils.py](../utils/audio_utils.py)
- [utils/data_pipeline.py](../utils/data_pipeline.py)

### Model
- [models/hybrid_model.py](../models/hybrid_model.py)

### Detection and Inference
- [inference/detector.py](../inference/detector.py)
- [inference/production_detector.py](../inference/production_detector.py)
- [production_api.py](../production_api.py)
- [app.py](../app.py)

### Configuration and Artifacts
- [config.py](../config.py)
- [paths.py](../paths.py)

## Repository Maturity Notes

The repository shows a substantial amount of implementation work, but the documentation is mixed between engineering usage and research methodology. The codebase is functional, but it is not yet fully organized around a single publication-grade experimental protocol.

## Executive Summary

- Repository maturity score: 7/10
- Software engineering score: 7/10
- Research readiness score: 5/10

### Major strengths
- Clear separation of training, calibration, evaluation, and inference.
- Multiple deployment interfaces.
- Existing artifact and reproducibility artifacts.
- Well-structured model and detector components.

### Major risks
- The repository mixes production deployment concerns with research experiment logic.
- The evaluation path appears to use a validation split rather than a fully separated test protocol.
- The current fusion and calibration logic are fixed and not yet condition-aware or learned.
- The repository contains several documentation files that describe methods, but the evidence for novelty remains to be experimentally validated.

### Recommended next actions
- Preserve the current architecture while clarifying the experimental protocol.
- Separate research experiments from deployment artifacts more explicitly.
- Add a transparent ablation and baseline framework.
- Formalize the publication claim around a single clearly defined contribution.
