# Normalization Protocol

## Current Protocol

The preprocessing path uses fixed mel normalization constants in [utils/audio_utils.py](../utils/audio_utils.py). Learned preprocessing statistics are not fit from validation or test rows; the active calibration path uses the train split only.

## Verified Observation

No per-dataset normalization statistics are learned from the training, validation, or test data.

## Impact

This is leakage-safe with respect to normalization fitting.
