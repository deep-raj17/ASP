# Phase 3 Implementation Changelog

## Test protection and evidence identity

- Added `utils/experiment_contract.py` with frozen-protocol validation,
  phase-aware split access, deterministic config serialization, canonical
  hashes, immutable run contracts, and prediction identity/completeness checks.
- `evaluate.py` rejects protected test access outside an explicitly authorized
  Phase 8.
- `scripts/run_baselines.py` and `scripts/audit_shortcuts.py` are
  validation-only during development.
- `scripts/recompute_metrics.py` no longer writes
  `reports/test_predictions.csv`; outputs are validation-named and create-only.

## Experimental validity

- Baseline feature exports now include stable sample ID, source recording,
  split, machine, and noise metadata.
- Global learned fusion now uses grouped five-fold out-of-fold validation
  predictions and train-fold normalization.
- Reliability-gate training accepts a disjoint selection loader.
- Added `scripts/run_reliability_cv.py` for grouped outer-fold predictions with
  disjoint inner-fold early stopping.
- Corrected the baseline report’s balanced-accuracy calculation.

## Numerical correctness and artifact safety

- Historical checkpoint loading uses restricted deserialization with only the
  project’s `EvalMetrics` class allowlisted.
- CUDA mixed precision uses BF16 when supported and otherwise FP32. FP16 was
  empirically non-finite on the preserved checkpoint.
- Production, evaluation, training, audits, and baseline extraction share the
  safe precision policy.

## Isolated execution

- `train.py --help` is now stateless.
- `train.py --submission-run` enforces frozen seeds and epoch limits, disables
  implicit resume, writes to a unique run directory, and creates an immutable
  run contract before training.
- Dry-run contract:
  `artifacts/submission_recovery/runs/phase3-contract-dryrun-20260727/run_contract.json`.
