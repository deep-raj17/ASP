# Phase 1 Repository State Capture

**Captured:** 2026-07-27 (Asia/Calcutta)  
**Scope:** read-only scientific-state inspection before new experiments

## Identity

| Field | Observed value |
|---|---|
| Repository root | `C:\ASP\ASP` |
| Branch | `blackboxai/research-integrity-audit` |
| HEAD | `3b78096e6ffcfb7f6ebff5fd6705f6b75124c2c7` |
| Worktree | DIRTY before Phase 1; extensive staged, unstaged, and untracked user work |
| Enumerated repository files | 636 excluding `.git` and `.venv`; 448 visible to `rg --files` |
| Repository file bytes | 26,750,422,761 bytes (25,511.19 MiB), dominated by checkpoints |
| Tags | `ros-cli-01-v1.0.0`; `ros-core-01-v1.0.0`; `ros-core-02-v1.0.0`; `ros-core-03-v1.0.0`; `ros-project-01-chaad-v1.0.0` |
| Remote | `origin` fetch/push: `https://github.com/deep-raj17/ASP` |

No clean, reset, commit, checkout, staging, or push operation was performed.
`GIT_STATUS_SNAPSHOT.txt` preserves the porcelain snapshot.

## Material state observations

- `EXP-CHAAD-001` is the only reconstructable full CHAAD training run.
- `checkpoints/best_model.pt` and its preserved copy have SHA-256
  `7d58293ea5138b730811bb261de0b971a8a5e4e52665961b48f2f4a4d85ad7e9`.
- The repository contains 100 epoch checkpoints plus the selected checkpoint;
  their existence is not evidence of independent reproducibility.
- The current manifest SHA-256 is
  `7c689508cbed4d49d05ec2891b315b27722ff01a8a62b6b1c4f610e3afcd0136`;
  the sidecar currently contains the same value. Earlier preserved provenance
  recorded a one-character mismatch, so the correction history remains
  unexplained even though current bytes agree.
- `reports/test_predictions.csv` contains 28,254 records for `id_00` and
  `id_02`, which are the current validation machines. Its filename is
  misleading and it is not evidence of protected-test evaluation.

## Phase 1 dry-run incident

`train.py --help` was attempted because the protocol permits non-destructive
help checks. The script has no CLI help guard and began ordinary startup. It
rewrote `artifacts/experiment_provenance.json` and created an 88-byte
TensorBoard event file before the output pipe terminated the process.

- No epoch completed.
- No model checkpoint changed; `best_model.pt` still matches the certified
  SHA-256 above.
- The overwritten provenance was restored byte-for-byte from its independently
  recorded migration inventory identity. Restored SHA-256:
  `fac6fa40c123caa48e8ab33d56149ceb07de71bcf43db1d55e9c83d6315aa459`.
- The empty-start event file is retained at
  `logs/events.out.tfevents.1785133442.nielit.34376.0` as audit evidence and is
  classified `NOT SCIENTIFIC EVIDENCE`.
- Because startup is stateful, `train.py` is classified as having no safe
  `--help`/dry-run mode.

## Freeze boundary

No dataset was written, no protected test evaluation was run, no training or
retraining was completed, no architecture was modified, and Phase 2 was not
started.
