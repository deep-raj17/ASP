# Technical Debt and Risks

## Critical

### 1. No fully separated test protocol
The repository appears to evaluate primarily on a validation split. This is a major issue for publication-grade claims because the final reported numbers may be optimistic if the threshold is also selected using validation data.

### 2. Fusion and calibration are fixed rather than learned
The current anomaly score is produced by a fixed weighted combination of calibrated scores. This is practical but limits the scientific novelty and the ability to adapt to condition-dependent reliability.

## High

### 3. Mixed research and production responsibilities
The repository combines model training, deployment, API serving, and export logic in the same codebase. This makes it harder to distinguish experimental workflows from production workflows.

### 4. Reproducibility is partially but not fully formalized
The repository has documentation and some artifact files, but a strict and machine-readable reproducibility protocol is not yet complete.

### 5. Limited evidence for novelty
The system uses multiple recognized methods, but the repository does not yet present a single clearly validated novel contribution with corresponding ablations.

## Medium

### 6. Some overlap in preprocessing and scoring logic
There is overlap between the standard detector and the production detector, and some preprocessing logic is repeated across entry points.

### 7. Hard-coded or implicit paths and configuration assumptions
The configuration is centralized, but the repository still relies on local dataset paths and environment-specific assumptions that are not fully abstracted.

## Low

### 8. Documentation is present but unevenly structured
The documentation is useful, but a more uniform structure would help a new researcher navigate the repository more quickly.
