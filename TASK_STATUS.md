# Task Status

Last updated: 2026-07-27

## Objective

Inspect the full repository, identify locally actionable correctness and
security issues, implement one focused task at a time, and verify every change.

## Preservation Constraints

- The working tree was already heavily dirty at session start.
- Existing staged, unstaged, and untracked content is user-owned.
- Do not install packages, retrain models, regenerate frozen artifacts, deploy,
  or access the held-out test split without explicit need and authorization.

## Completed

- Read the project context in the order mandated by `AGENTS.md`.
- Inventoried the CHAAD ML stack and ROS workflow-control stack.
- Read dependencies, configuration, core implementation, tests, and issue logs.
- Captured the initial Git state.
- Parsed 100 Python files successfully after the focused change.
- Ran the complete discovered test suite: 73 tests passed.
- Ran `pip check`: seven dependency conflicts were reported.
- Verified `checkpoints/best_model.pt` exists.
- Verified required calibration and advertised FP16/ONNX artifacts are missing.

## Current Work

### Task 1 — Fail-closed inference artifact loading

Status: VERIFIED

- Production UI, API, and smoke verification reject missing required artifacts.
- Model and calibration loading use restricted PyTorch deserialization.
- Three focused regression tests cover the shared artifact contract.

## Test Results

| Command | Result |
|---|---|
| Python AST parse over repository | VERIFIED: 100 parsed, 0 failures |
| `python -m pytest tests/test_inference_artifacts.py -p no:cacheprovider -q` | VERIFIED: 3 passed |
| `python -m pytest tests -p no:cacheprovider -q` | VERIFIED: 73 passed |
| `python scripts/verify_inference.py` | BLOCKED as designed: calibration missing |
| `python -m pip check` | FAILED: 7 environment dependency conflicts |

## Unresolved Problems

- Detector calibration is missing, so calibrated production inference is
  BLOCKED even though the primary checkpoint exists.
- FP16 and ONNX artifacts advertised by the README are missing.
- Dataset acquisition provenance remains BLOCKED.
- Multi-run and cross-hardware reproducibility remain UNVERIFIED.
- Dependency metadata and the active environment are inconsistent.
- Publication experiments requiring retraining or new held-out test use are
  outside this locally safe task.

## Next Action

Restore or regenerate detector calibration only under explicit authorization,
then rerun `python scripts/verify_inference.py` for a real forward-pass check.
