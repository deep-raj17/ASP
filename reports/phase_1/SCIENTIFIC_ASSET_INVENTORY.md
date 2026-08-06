# Scientific Asset Inventory — Interpretation

The companion CSV is the complete file-level inventory for the repository at
Phase 1 freeze time. It includes source, configuration, checkpoints, prediction
exports, metrics, logs, reports, tests, environment specifications, governance
records, provenance records, figures, and temporary/failed outputs. `.git` and
`.venv` internals are excluded because they are not project assets.

## Classification policy

| Status | Phase 1 interpretation |
|---|---|
| `VALIDATED` | Directly checked against lower-level evidence during this or a preserved audit |
| `PROVISIONALLY USABLE` | Useful for diagnosis or protocol design, but not final publication evidence |
| `INCOMPLETE` | Intended output exists without all required companions or metadata |
| `CORRUPTED` | Known identity/content defect makes the artifact invalid for its intended claim |
| `DUPLICATED` | Byte-identical or functionally duplicated material; retain but avoid double counting |
| `STALE` | Reflects a superseded protocol, state, or metric |
| `UNVERIFIED` | Exists, but execution or scientific correctness is not established |
| `NOT SCIENTIFIC EVIDENCE` | Governance, narrative, cache, temporary, or operational record that cannot itself support an empirical claim |

Status assignment is intentionally conservative. Executable code is generally
`UNVERIFIED` unless a direct audit validates its relevant behavior. Narrative
documents are generally `NOT SCIENTIFIC EVIDENCE` even when useful, because
they must resolve to machine-readable artifacts.

## Most important assets

- `artifacts/EXP-CHAAD-001/checkpoint.pt`: `PROVISIONALLY USABLE`; byte identity
  with `checkpoints/best_model.pt` is certified, but the run is underfit and
  incompletely reproducible.
- `artifacts/EXP-CHAAD-001/validation_predictions_corrected.csv`: `VALIDATED`;
  28,254 unique validation identities, valid labels, finite scores, correct
  split membership, and independently recomputable metrics.
- `artifacts/EXP-CHAAD-001/validation_predictions.csv`: `CORRUPTED`; 30
  duplicated sample IDs, 60 affected rows.
- `reports/test_predictions.csv`: `PROVISIONALLY USABLE` only as an older
  validation export; its `test` filename contradicts its `id_00`/`id_02`
  contents and must never be cited as protected-test evidence.
- `metadata/dataset_manifest.csv`: `VALIDATED` for current bytes, schema,
  counts, split assignment, and full-corpus audit linkage. Official archive
  lineage remains unresolved.
- `models/reliability.py`: `UNVERIFIED` as a scientific contribution. It
  imports and is technically coherent, but no qualifying ablation or paired
  effectiveness experiment exists.
- `artifacts/baselines/diagnostic_baselines.json`: `PROVISIONALLY USABLE`;
  useful diagnosis, not publication-grade baseline evidence.
- `scripts/run_baselines.py` and `scripts/statistical_validation.py`:
  `UNVERIFIED`; CLI help works, but script existence is not execution evidence.
- 100 epoch checkpoints: retained, but only the epoch-6 selected checkpoint is
  registered for scientific reuse. The others are primarily trace/history
  artifacts and do not represent independent experiments.
- Phase 1 reports and ROS/PMPS governance records: `NOT SCIENTIFIC EVIDENCE`
  unless they cite a lower-level artifact.

## Conflicts frozen by this inventory

1. Current manifest and sidecar bytes match, while an older preservation record
   says they did not. Current integrity is validated; change provenance is not.
2. The raw archive comparison has eight `MATCH`, one `MISMATCH`, and three
   `LOCAL_ARCHIVE_MISSING` rows. Some later narratives say five matches; those
   narratives are stale on this count.
3. The baseline report displays PR-AUC values that disagree with
   `diagnostic_baselines.json`; the machine-readable JSON takes precedence.
4. The old near-perfect evaluation report is contradicted by corrected
   identity-safe validation predictions.

The CSV hash column is reproducible from the exact file bytes present when the
inventory was generated. It is a repository snapshot, not a certification that
every file is scientifically valid.
