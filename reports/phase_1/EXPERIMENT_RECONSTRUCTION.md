# Experiment Reconstruction

## EXP-CHAAD-001

`EXP-CHAAD-001` is **COMPLETE BUT PROVISIONAL**. It is reconstructable as a
historical run, but not independently reproducible as an exact rerun.

### Reconstructed protocol

- Architecture: EfficientNet-B4 convolutional backbone, Transformer temporal
  module, autoencoder branch, supervised contrastive representation objective,
  and multi-signal anomaly scoring.
- Dataset manifest:
  `metadata/dataset_manifest.csv`, SHA-256
  `7c689508cbed4d49d05ec2891b315b27722ff01a8a62b6b1c4f610e3afcd0136`.
- Split: train `id_04` (12,045), validation `id_00` + `id_02` (28,254),
  protected test `id_06` (12,747).
- Seed: 42.
- Run-start time: `2026-07-21T12:05:03.484644+00:00`.
- Reported command: `python train.py`; no original shell transcript.
- Optimisation: AdamW, learning rate `1e-4`, batch size 32, 100 epochs,
  OneCycle schedule, mixed precision and gradient accumulation 2 according to
  the active preservation snapshot. The complete exact training-time config
  was not captured.
- Loss weights: BCE 1.0, contrastive 0.3, reconstruction 0.05; positive-class
  weight 5.0.
- Checkpoint rule: strictly lower validation loss. Epoch 6 was selected at
  validation loss 2.2707729198; the highest logged validation ROC-AUC was epoch
  11 (0.6176229715), which was not selected.
- Final epoch metrics were worse: validation loss 3.7270186234 and ROC-AUC
  0.5232595824.
- Selected checkpoint SHA-256:
  `7d58293ea5138b730811bb261de0b971a8a5e4e52665961b48f2f4a4d85ad7e9`.
- Corrected selected-checkpoint validation ROC-AUC: 0.6002609445.

### Training conclusion

The training audit found finite, nonconstant batches and valid gradient flow,
but only partial tiny-batch overfitting and worsening validation loss. The
frozen classification is **MODEL UNDERFITTING**. This is not proof that the
pipeline is broken, but it is not a strong final model.

### Provenance limitations

The Git commit is recorded, but the exact dirty-tree patch is not. The precise
shell command, full training-time config, GPU identity, OS/driver/cuDNN, and
dependency lock are incomplete. The current interpreter differs from the
training interpreter. Official dataset byte lineage is conflicting.

## Derived exports and diagnostic runs

- The original EXP-CHAAD-001 validation export is **CORRUPTED** because identity
  collisions create 30 duplicated IDs and 60 affected rows.
- The corrected validation export is a valid derived audit artifact. It does
  not convert the underlying training run into a final result.
- Diagnostic classical baselines are **COMPLETE BUT PROVISIONAL**. They suggest
  a difficult task and only a small CHAAD advantage, but the sampled input
  rows, exact matched protocol, and publication-grade pairing are absent. The
  baseline narrative also conflicts with its JSON on PR-AUC values.
- Publication baseline, ablation, and statistical-validation work are
  **SCRIPT ONLY — NOT EXECUTED**.

No artifact proves a protected-test evaluation under the current protocol.
The filename `reports/test_predictions.csv` must not be interpreted as such:
its 28,254 rows use validation IDs `id_00` and `id_02`.
