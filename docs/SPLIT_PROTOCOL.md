# Split Protocol

## Current Protocol

The repository now uses a manifest-backed split protocol with explicit train, validation, and test assignments. The shared interface in [utils/split_utils.py](../utils/split_utils.py) loads the authoritative manifest and validates split membership before returning rows to the active training and evaluation loaders.

## Verified State

The regenerated manifest currently reports:

- train: 12,045 records
- validation: 28,254 records
- test: 12,747 records

Machine IDs are disjoint across splits.

## Impact

This makes the active data-loading path consistent with the manifest and prevents independent split recreation in the training and calibration code paths.
