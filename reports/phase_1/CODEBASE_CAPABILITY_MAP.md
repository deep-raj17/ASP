# Codebase Capability Map

## Verification performed

- Parsed all 99 repository Python files with `ast.parse`: **99 passed, 0
  syntax failures**.
- Imported ten central modules with `python -B`: `config`, `data.dataset`,
  `utils.split_utils`, `utils.audio_utils`, `models.hybrid_model`,
  `models.reliability`, `training.loss`, `training.trainer`,
  `inference.detector`, and `utils.metrics`: **10 passed, 0 import failures**.
- `evaluate.py`, `scripts/run_baselines.py`,
  `scripts/statistical_validation.py`, and
  `scripts/run_publication_audit.py` expose working help output.
- `train.py` does **not** expose a safe help path; `--help` enters normal
  startup. The incident and exact restoration are documented in
  `REPOSITORY_STATE_CAPTURE.md`.

These checks validate parse/import/CLI availability only. They do not validate
scientific effectiveness.

| Capability | Implementation / entry point | Inputs and configuration | Outputs | Connected / execution evidence | Phase 1 assessment |
|---|---|---|---|---|---|
| Dataset loading | `data/dataset.py` | `cfg.data.dataset_dir`, manifest, audio config | tensors, labels, metadata | Connected; preserved training and audits | Implemented; current split consumer requires manifest |
| Manifest loading | `utils/split_utils.py` | `metadata/dataset_manifest.csv` | normalized split rows | Connected and audited | VALIDATED for current manifest |
| Split enforcement | `data/dataset.py`, `utils/split_utils.py` | machine-independent manifest | train/val/test loaders | 12,045/28,254/12,747 audited | VALIDATED within documented scope |
| Preprocessing | `utils/audio_utils.py` | 16 kHz, 10 s, mel settings | normalized mel tensors | Training/data audit evidence | PROVISIONALLY USABLE |
| Augmentation | `data/dataset.py`, config | training samples/config | augmented tensors | Active config exists; exact run-time config incomplete | UNVERIFIED as an effectiveness factor |
| Feature extraction | `models/hybrid_model.py` | mel tensors, model config | embeddings/features | Imports and checkpoint exist | PROVISIONALLY USABLE |
| Model construction | `models/hybrid_model.py`, `train.py` | model config | hybrid model | One completed provisional run | Implemented; effectiveness weak |
| Reliability-aware fusion | `models/reliability.py` | anomaly signals plus condition metadata | sample-dependent weights/fused score | Imports; no qualifying experiment | Implemented, scientifically UNVERIFIED |
| Loss computation | `training/loss.py` | logits, embeddings, reconstruction, labels | BCE/SupCon/reconstruction loss | Training audit and gradients | Functional; weighting implicated in underfitting |
| Optimisation | `training/trainer.py` | AdamW/config | epoch checkpoints/logs | 100-epoch history exists | Functional for one provisional run |
| Checkpointing | `training/trainer.py`, `utils/checkpoint.py` | validation loss | epoch and best checkpoints | 100 epoch files; epoch 6 selected | Byte identity validated; selection minimizes val loss |
| Prediction export | `evaluate.py`, `scripts/audit_evaluation_pipeline.py` | checkpoint and split loader | prediction CSV | Original corrupted; corrected export validated | Corrected audit path VALIDATED |
| Test-set protection | `evaluate.py`, `artifacts/final_test_lock.json` | frozen threshold metadata | guarded evaluation | Help shows explicit split; no protected-test execution in Phase 1 | Policy implemented; final use not yet exercised |
| Threshold selection | `utils/metrics.py` | validation labels/scores | Youden threshold | Independently recomputed | VALIDATED for validation, not final test |
| Metric computation | `utils/metrics.py`, `scripts/recompute_metrics.py` | identities, labels, scores | ROC/PR/EER/classification metrics | Corrected predictions independently recomputed | VALIDATED for audited validation export |
| Baseline execution | `scripts/run_baselines.py` | dataset/config/checkpoint | comparison JSON | Help works; publication suite not executed | SCRIPT ONLY — NOT EXECUTED |
| Diagnostic baselines | `scripts/run_diagnostic_baselines.py` | sampled features | diagnostic JSON | Output exists | COMPLETE BUT PROVISIONAL; insufficient provenance |
| Ablation support | narrative plan; no dedicated runner found | future frozen factors | none | No execution evidence | INCOMPLETE |
| Statistical analysis | `scripts/statistical_validation.py` | candidate/baseline prediction CSVs | bootstrap/test report | Help works; final paired inputs absent | SCRIPT ONLY — NOT EXECUTED |
| Figure generation | training-audit plotting path | TensorBoard scalars | PNG curves | Four training curves exist | Reusable diagnostically; regenerate paper figures |
| Production inference | `inference/production_detector.py` | calibrated model | detector outputs | no end-to-end evidence | UNVERIFIED |
| Edge deployment | `edge_deploy/` | export/model/target device | ONNX/device artifacts | no target-hardware evidence | UNVERIFIED; not critical path |

## Obsolete or hazardous paths

- The old file-level/machine-dependent split narrative is superseded by the
  manifest-based machine-independent protocol.
- `reports/test_predictions.csv` is semantically mislabeled.
- `train.py` startup is stateful even with `--help`; future dry-run support must
  be designed explicitly before it is used in an audit.
- Existing scripts are capability evidence only, never proof that their planned
  experiments ran.
