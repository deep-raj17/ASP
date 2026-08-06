# Data Integrity Report

## Executive Summary

The repository now uses a manifest-based, machine-independent split protocol with an explicit test split. The active dataset loaders are wired through a shared manifest interface so training, validation, calibration, and evaluation consume the same authoritative split assignments.

## Verified Findings

- The manifest contains 53,046 records.
- The active manifest-backed splits are train=12,045, validation=28,254, and test=12,747.
- The shared split loader validates train/validation/test membership and rejects unknown split names.
- The active dataset loaders consume rows from the manifest-defined train and validation splits.
- The calibration normal loader uses normal train rows only.
- The regression suite passes with 29 tests.

## Verification Gates

- Gate A — Manifest contains train/validation/test: PASS.
- Gate B — Machine IDs are disjoint: PASS.
- Gate D — Training consumes train only: PASS.
- Gate E — Validation consumes validation only: PASS.
- Gate F — Normalization fitted without test data: PASS.
- Gate G — Calibration excludes test data: PASS.
- Gate K — End-to-end regression suite passes: PASS.

## Remaining Gaps

- Threshold selection and final test evaluation remain validation-focused and do not yet expose a fully frozen test-only evaluation mode.
- A read-only final-test evaluation workflow is still pending.
