# Data Flow

## End-to-End Flow

The repository processes audio data through the following pipeline:

1. Audio files are discovered from the dataset root.
2. Files are assigned to train or validation splits.
3. Waveforms are loaded and preprocessed.
4. Log-mel spectrograms and MFCCs are extracted.
5. The model produces embeddings, logits, and reconstruction outputs.
6. Anomaly scores are computed from reconstruction, embedding, Mahalanobis, and contrastive signals.
7. Scores are calibrated and fused.
8. The final score is thresholded and exposed through evaluation or inference.

## Stage-by-Stage Details

### 1. Dataset Discovery
- Implemented in [data/dataset.py](../data/dataset.py)
- Scans the MIMII tree for WAV files, parses labels from directory structure, and assigns train/validation splits.

### 2. Audio Loading
- Implemented in [data/dataset.py](../data/dataset.py)
- Loads waveform with soundfile, converts to mono, resamples if needed, pads or trims to the target duration.

### 3. Feature Extraction
- Implemented in [utils/audio_utils.py](../utils/audio_utils.py)
- Produces log-mel spectrograms, MFCCs, and a waveform branch.

### 4. Augmentation
- Implemented in [utils/audio_utils.py](../utils/audio_utils.py)
- Applies stochastic waveform noise and SpecAugment when augmentation is enabled.

### 5. DataLoader and Batch Formation
- Implemented in [data/dataset.py](../data/dataset.py)
- Builds train and validation DataLoader objects.

### 6. Model Forward Pass
- Implemented in [models/hybrid_model.py](../models/hybrid_model.py)
- Produces embeddings, logits, reconstruction, attention weights, and pooled features.

### 7. Scoring and Calibration
- Implemented in [inference/detector.py](../inference/detector.py)
- Computes anomaly signals and normalizes them via z-score calibration.

### 8. Thresholding and Output
- Implemented in [inference/detector.py](../inference/detector.py) and [inference/production_detector.py](../inference/production_detector.py)
- Converts the fused score to a 0-1 anomaly decision.

## Notable Data-Flow Observations

- The pipeline is explicit and understandable.
- The dataset split is deterministic but currently does not appear to implement a fully independent test set.
- The calibration step is based on normal training samples only.
- The final decision is based on a fixed weighted fusion strategy rather than a learned reliability mechanism.
