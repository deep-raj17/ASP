# Inference Pipeline

## Overview

Inference is implemented in multiple paths:
- a standard detector in [inference/detector.py](../inference/detector.py),
- a production-optimized detector in [inference/production_detector.py](../inference/production_detector.py),
- a REST API in [production_api.py](../production_api.py),
- and a Gradio UI in [app.py](../app.py).

## Standard Detector Flow

1. The model receives a mel spectrogram.
2. The model outputs embeddings, logits, reconstruction, attention weights, and pooled features.
3. The detector computes:
   - reconstruction error,
   - embedding distance,
   - Mahalanobis distance,
   - contrastive distance.
4. Each signal is z-score calibrated using normal-data statistics.
5. The scores are sigmoid-mapped and fused with fixed weights.
6. The final score is thresholded and converted into a label and health/risk output.

## Production Detector Flow

The production detector follows the same scoring logic but adds:
- batch processing,
- optimized inference settings,
- and production-oriented output generation.

## API and UI

- [production_api.py](../production_api.py) exposes inference through Flask endpoints.
- [app.py](../app.py) exposes inference through a Gradio interface.

## Notes

The inference path is operational and fairly complete. However, the current scoring logic is fixed-weight and not yet adaptive or condition-aware. That is a relevant limitation for both engineering robustness and research novelty.
