# Phase 1 Repository State

Captured: 2026-07-27  
Repository: `C:\ASP\ASP`  
Branch: `blackboxai/research-integrity-audit`  
HEAD: `3b78096e6ffcfb7f6ebff5fd6705f6b75124c2c7`

## State

The working tree is heavily dirty with staged, unstaged, and untracked work.
The authoritative snapshot is the Phase 1 entry in the session transcript and
the earlier frozen capture at
`reports/phase_1/GIT_STATUS_SNAPSHOT.txt`. No pre-existing change was reverted,
staged, committed, or overwritten during this reconstruction.

## Runtime

| Component | Observed value |
|---|---|
| Python | 3.10.11 |
| PyTorch | 2.5.1+cu121 |
| torchvision | 0.20.1+cu121 |
| torchaudio | 2.5.1+cu121 |
| CUDA available | yes |
| CUDA runtime | 12.1 |
| GPU | NVIDIA GeForce RTX 4070 SUPER |
| pandas | 2.3.3 |
| scikit-learn | 1.7.2 |
| NumPy | 1.26.4 |

`requirements.txt` describes the CHAAD ML environment. `pyproject.toml`
describes the separate ROS workflow/CLI package and requires Python 3.10+.
`pip check` reports seven conflicts in the active shared environment; see
`TASK_STATUS.md`.

## Architecture

CHAAD is a Python/PyTorch acoustic-anomaly system:

1. manifest-driven MIMII dataset loading;
2. log-mel preprocessing;
3. EfficientNet CNN;
4. Transformer or BiLSTM temporal encoder;
5. attention pooling;
6. classifier, contrastive embedding, and autoencoder branches;
7. calibrated reconstruction, embedding, Mahalanobis, and contrastive scores;
8. fixed or reliability-aware fusion;
9. Gradio, Flask, and edge inference surfaces.

The repository also contains a ROS evidence/workflow/registry/CLI subsystem.
That subsystem controls research state but is not empirical model evidence.

## Dataset and split identity

The source of truth is `metadata/dataset_manifest.csv`.

| Split | Machine IDs | Rows |
|---|---|---:|
| train | id_04 | 12,045 |
| validation | id_00, id_02 | 28,254 |
| protected test | id_06 | 12,747 |

The extracted corpus is internally consistent relative to the manifest.
Official archive lineage remains BLOCKED by conflicting/missing acquisition
evidence. `E:\MIMII` was not read or modified in this phase.

## Evidence boundary

- Safe validation evidence was recomputed from
  `artifacts/EXP-CHAAD-001/validation_predictions_corrected.csv`.
- The original validation export is corrupted by duplicate sample identities.
- `reports/test_predictions.csv` was not opened in this recovery phase. Earlier
  frozen QA classifies it as misleadingly named validation content; it remains
  excluded from publication.
- No protected test evaluation occurred.

## Detailed frozen evidence

The exhaustive 656-row hashed inventory remains at
`reports/phase_1/SCIENTIFIC_ASSET_INVENTORY.csv`. This recovery package indexes
that evidence rather than regenerating or overwriting it.
