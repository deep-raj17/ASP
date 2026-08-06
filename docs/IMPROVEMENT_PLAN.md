# Improvement Plan

## Immediate Priorities

1. Formalize the experimental protocol
   - Create a clearly separated train/validation/test split.
   - Freeze the threshold on validation and evaluate on test data.

2. Strengthen the scientific contribution
   - Define one primary novelty claim and test it with ablations.
   - Avoid presenting the system as novel merely because it combines known methods.

3. Clarify artifact management
   - Keep training artifacts, evaluation artifacts, and deployment artifacts clearly separated.

4. Add stronger baseline comparisons
   - Compare the current pipeline against simpler baselines such as single-score detection and fixed-weight fusion.

5. Improve reproducibility documentation
   - Make the exact experiment recipe, seed, and artifact paths explicit.

## Medium-Term Improvements

- Introduce a condition-aware or reliability-aware fusion module for the anomaly scores.
- Add subgroup analysis by machine type, machine ID, and noise condition.
- Prepare a standardized evaluation report that can be used for publication.

## Long-Term Improvements

- Separate research experiments from production deployment code more explicitly.
- Add automated regression tests for training, calibration, and inference entry points.
- Standardize metrics and reporting outputs across scripts.
