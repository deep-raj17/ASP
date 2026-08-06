# Calibration Split Policy

## Policy

Calibration statistics are fitted using only normal samples from the manifest-defined train split.

- Model fitting: train only
- Calibration-statistic fitting: train-normal only
- Validation: never used for calibration fitting
- Test: never used for calibration fitting

## Implementation

The active calibration path uses [data/dataset.py](../data/dataset.py) through get_normal_loader, which builds a loader from the train split only. The shared manifest interface in [utils/split_utils.py](../utils/split_utils.py) is the single authority for split selection.
