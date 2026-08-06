# EXP-CHAAD-001

**Status:** OFFICIAL PROVISIONAL RESULT  
**Classification:** Provisional - pipeline not yet independently verified  
**Date:** 2026-07-23T09:30:00+05:30

## Overview

This is the first official CHAAD training run under the machine-independent split protocol. The model shows near-random performance (ROC-AUC = 0.5233), indicating either a pipeline issue, a harder generalization problem, or genuine model failure.

## Performance

- **Selected checkpoint:** epoch 6 (minimum validation loss 2.270773)
- **Selected-checkpoint ROC-AUC:** 0.600298
- **Highest logged ROC-AUC:** 0.617623 at epoch 11 (not selected)
- **Final validation ROC-AUC:** 0.523260
- **Final validation balanced accuracy:** 0.543263
- **Final validation EER:** 0.457360
- **Final train loss:** 1.050219
- **Final validation loss:** 3.727019

Checkpoint selection minimizes validation loss. The preserved best checkpoint is
therefore epoch 6 and differs from the final epoch (100).

## Artifacts

- `checkpoint.pt` - Model checkpoint (SHA-256: 7d58293ea5138b730811bb261de0b971a8a5e4e52665961b48f2f4a4d85ad7e9)
- `checkpoint.sha256` - Checkpoint checksum
- `config_snapshot.json` - Configuration snapshot
- `provenance.json` - Experiment provenance
- `git_commit.txt` - Git commit hash
- `manifest_checksum.txt` - Dataset manifest checksum
- `environment.json` - Environment specification
- `training_summary.json` - Training summary
- `training_command.txt` - Training command
- `validation_metrics.json` - Validation metrics

## Provenance

- **Git Commit:** 686c450bd416f6cf921befe4156d1a27b26105c2
- **Git Branch:** blackboxai/research-integrity-audit
- **Dataset Manifest:** metadata/dataset_manifest.csv
- **Calculated Dataset Manifest SHA-256:** 7c689508cbed4d49d05ec2891b315b27722ff01a8a62b6b1c4f610e3afcd0136
- **Repository sidecar checksum:** 7c689508cbed4d49d05ec2891b315b27722ff01c8a62b6b1c4f610e3afcd0136 (**MISMATCH**)
- **Split Protocol:** machine_independent
- **Random Seed:** 42

## Reproducibility Assessment

**PARTIALLY REPRODUCIBLE.** The model state, optimizer/scheduler/scaler state,
seed, manifest hash, selected configuration fields, and TensorBoard history are
available. Exact reproduction is not currently established because the training
ran from a dirty working tree whose patch was not preserved, the complete
training-time configuration and exact shell command were not captured, the
active Python interpreter now differs from the recorded training interpreter,
and no exact dependency/driver lock is available.

## Notes

- The checkpoint is preserved from the original location (copy, not move)
- Source and copied checkpoint hashes were verified identical on 2026-07-24
- Training logs contain 100 validation epochs in TensorBoard format
- Exact original shell command remains unresolved
- The full active configuration snapshot was captured at preservation time; only a subset was captured at training start
- `metadata/dataset_manifest.sha256` does not match the current manifest bytes and was deliberately left unchanged for audit visibility
- No test set evaluation performed
