# Calibration Protocol

## Current Protocol

Calibration is fitted from the normal training subset using the detector reference statistics in [inference/detector.py](../inference/detector.py). The active path is documented in [docs/CALIBRATION_SPLIT_POLICY.md](CALIBRATION_SPLIT_POLICY.md).

## Verified Observation

The intended protocol is train-normal-only calibration, which is consistent with leakage-safe calibration practice.

## Impact

This component does not appear to leak validation or test data into the calibration stage.
