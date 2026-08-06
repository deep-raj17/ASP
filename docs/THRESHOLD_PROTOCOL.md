# Threshold Protocol

## Current Protocol

The repository evaluates validation metrics and currently does not implement a separate frozen threshold-selection workflow for final test evaluation. The active pipeline remains validation-focused and the shared manifest loader is the authoritative split source.

## Verified Observation

The threshold protocol is therefore not yet publication-grade for a frozen-threshold test evaluation.

## Required Correction

Select the threshold on validation data only, freeze it, and evaluate it on a separate test set.
