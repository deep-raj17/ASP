# Architecture Decision Records — CHAAD Project

> Reconstructed from source code, configuration, documentation, Git history, and audit artifacts. See `docs/DECISIONS.md` evidence rules in `AGENTS.md`.

---

## ADR-ROS-001: Missing PMPS Evidence Maps to BLOCKED

- **Date**: 2026-07-24
- **Status**: Accepted
- **Decision**: Preserve the historical binary PMPS-01 `FAIL` artifact, but
  evaluate missing or unevaluable required evidence as `BLOCKED`. Reserve
  `FAILED` for a defined execution failure and `UNSATISFIED` for evaluated
  evidence that fails a requirement.
- **Rationale**: Missing acquisition provenance does not prove dataset
  corruption or model failure.
- **Effect**: PMPS 1.0.0 contains 13 machine-readable PMPS-01 requirements.
  The full-corpus audit satisfies current-file integrity; local release identity
  remains blocked pending authoritative archive provenance.
- **Evidence**: `migration/chaad/PMPS_01_EVIDENCE_MATRIX.csv` and
  `migration/chaad/CHAAD_STATE_REPORT.md`.

---

## ADR-001: Machine-Independent Split Protocol

- **Decision ID**: ADR-001
- **Date**: 2026-07 (inferred from audit branch creation)
- **Status**: Accepted
- **Context**: The original split used `split_seed + relative_path` hashing, which could distribute samples from the same physical machine across train and val. This risks data leakage if recordings from the same machine share acoustic characteristics.
- **Decision**: Adopt a **machine-independent** protocol. Each machine ID (`id_00`, `id_02`, `id_04`, `id_06`) is assigned to exactly one split via `hash(machine_id) % 3`.
- **Alternatives considered**: 
  - File-hash split (rejected — does not guarantee machine isolation)
  - Stratified by machine_type + noise_condition (rejected — more complex, no clear benefit over per-ID isolation)
- **Rationale**: Machine-ID isolation is the strongest guarantee against leakage for this dataset. Each ID represents a physically distinct machine.
- **Implementation effect**: `data/dataset.py` now uses `split_utils.load_manifest_split()` instead of inline hashing. The manifest is the single source of truth.
- **Scientific effect**: No cross-split machine-ID leakage. Performance claims are defensible.
- **Source evidence**: `utils/split_utils.py` lines 49-75 (machine-ID overlap check), `_audit_check.py` Phase 3-4, `artifacts/research_audit/machine_split_table.csv`

---

## ADR-002: Three-Way Train/Validation/Test Split

- **Decision ID**: ADR-002
- **Date**: 2026-07 (inferred)
- **Status**: Accepted
- **Context**: The original code only had train/val splits. The `evaluate.py` script evaluated on validation and selected the threshold on validation — this conflates model selection with evaluation.
- **Decision**: Adopt a three-way split: train (id_04), validation (id_00 + id_02), test (id_06). Threshold is selected on validation via Youden's J. Final metrics are computed on test with the frozen threshold.
- **Alternatives considered**: 
  - Train/val only (rejected — no independent test evaluation)
  - K-fold cross-validation (rejected — machine-ID isolation becomes complex)
- **Rationale**: Three-way split is the standard for ML research. It separates model training, hyperparameter/threshold selection, and final evaluation.
- **Implementation effect**: `evaluate.py` supports `--split test` with frozen threshold from metadata. `artifacts/threshold_metadata.json` records the selection split.
- **Scientific effect**: Reported metrics are from an untouched test set. No test-data leakage into threshold selection.
- **Source evidence**: `evaluate.py` lines 60-67, `utils/metrics.py` `persist_threshold_metadata()`, `artifacts/threshold_metadata.json`

---

## ADR-003: Validation-Only Threshold Selection

- **Decision ID**: ADR-003
- **Date**: 2026-07 (inferred)
- **Status**: Accepted
- **Context**: The anomaly detection pipeline requires a decision threshold. Selecting it on test data would inflate reported metrics.
- **Decision**: Threshold is selected on the **validation split only** using Youden's J statistic (maximizes sensitivity + specificity - 1). The threshold metadata is persisted and reloaded for test evaluation.
- **Alternatives considered**: 
  - Fixed 0.5 threshold (rejected — not optimal for imbalanced data)
  - Percentile-based threshold (rejected — less principled than Youden's J)
- **Rationale**: Youden's J is a standard, threshold-agnostic method that gives equal weight to sensitivity and specificity.
- **Implementation effect**: `utils/metrics.py` `select_threshold()` and `persist_threshold_metadata()`. `evaluate.py` loads frozen threshold for test.
- **Source evidence**: `utils/metrics.py` lines 97-100, `evaluate.py` lines 76-78

---

## ADR-004: Train-Normal Calibration

- **Decision ID**: ADR-004
- **Date**: 2026-05 (from calibration_report.json timestamp)
- **Status**: Accepted
- **Context**: The anomaly detector needs reference statistics (μ, σ per signal, reference embeddings, covariance matrix) to compute meaningful z-scores and distances.
- **Decision**: All calibration statistics are fitted on **train_normal only** (normal samples from the training split). No abnormal samples, validation samples, or test samples are used.
- **Alternatives considered**: 
  - Full training set including abnormal samples (rejected — would contaminate the "normal" reference)
  - Validation set (rejected — would leak validation information into scoring)
- **Rationale**: The anomaly detector's reference distribution should represent "normal" operation. Using only train_normal ensures no leakage.
- **Implementation effect**: `calibrate.py` uses `get_normal_loader()` which filters to label=0 on train split. `inference/detector.py` `fit_reference_distribution()`.
- **Source evidence**: `calibrate.py` lines 40-98, `data/dataset.py` `get_normal_loader()`, `artifacts/research_audit/calibration_audit.json`

---

## ADR-005: Deterministic Execution and Seed Control

- **Decision ID**: ADR-005
- **Date**: 2026-07-21
- **Status**: Accepted
- **Context**: The original code had no explicit seed control beyond `split_seed=42`. Model initialization, data augmentation, and CUDA operations were non-deterministic.
- **Decision**: Add comprehensive seed control via `utils/seed.py`. Set `random_seed=42` and `deterministic_cudnn=True` in `TrainingConfig`. Record provenance on every run.
- **Alternatives considered**: 
  - No seed control (rejected — irreproducible results)
  - Seed-only without cuDNN determinism (rejected — CUDA non-determinism would cause variance)
- **Rationale**: Reproducibility requires controlling ALL random sources. cuDNN determinism trades speed for exact reproducibility.
- **Implementation effect**: `utils/seed.py` sets Python, NumPy, PyTorch, CUDA, and cuDNN seeds. `train.py` calls `set_seed()` and writes provenance JSON.
- **Scientific effect**: Two independent runs with identical seed and hardware should produce identical results.
- **Source evidence**: `utils/seed.py`, `config.py` lines 106-108, `train.py` lines 30-50

---

## ADR-006: Reliability-Aware Fusion as Primary Novel Contribution

- **Decision ID**: ADR-006
- **Date**: 2026-07-21
- **Status**: Accepted
- **Context**: The original system combined four anomaly signals using fixed hand-selected weights (0.30/0.25/0.30/0.15). This does not adapt to different machine types or noise conditions.
- **Decision**: The primary novel contribution is a **reliability-aware fusion module** (`models/reliability.py`) that learns sample-dependent weights conditioned on the embedding, machine type, and noise condition. Weights are trained on validation data using a pairwise ranking loss.
- **Alternatives considered**: 
  - Global learned weights (rejected — sample-independent, does not exploit condition information)
  - Equal-weight fusion (rejected — naive, ignores signal quality variation)
  - Fixed hand-selected weights (rejected — current baseline, no adaptation)
- **Rationale**: The novelty claim is defensible: the gate is sample-dependent, condition-aware, and trained on validation data only (no test leakage). Ablations can isolate each component.
- **Implementation effect**: `models/reliability.py` (ReliabilityGate + ReliabilityGateTrainer + MetadataMapper). Future: ablation study runner comparing all variants.
- **Scientific effect**: If the ablation study shows statistically significant improvement over fixed weights, the claim is supported.
- **Source evidence**: `docs/NOVELTY_AND_CONTRIBUTIONS.md`, `models/reliability.py`, `inference/detector.py` (fixed weights as baseline)

---

## ADR-007: Legacy Metrics Treatment

- **Decision ID**: ADR-007
- **Date**: 2026-07-21
- **Status**: Accepted
- **Context**: `checkpoints/eval_report.json` reports ROC-AUC 99.99997%, PR-AUC 99.99986% on the VALIDATION split. These were computed before the current split protocol was formalized.
- **Decision**: All legacy metrics are classified as **LEGACY — SUPERSEDED**. They cannot be cited as final results. Only metrics computed on the test split with a frozen validation threshold and documented provenance are considered valid.
- **Alternatives considered**: 
  - Treating legacy metrics as valid (rejected — unknown split protocol, no test set)
  - Deleting legacy reports (rejected — historical value for audit trail)
- **Rationale**: Scientific integrity requires knowing what evaluation protocol produced each number. The current protocol is different from the legacy one.
- **Implementation effect**: Legacy reports remain in `checkpoints/` and `artifacts/pre_validation_backup/` with caveats in documentation.
- **Source evidence**: `docs/CURRENT_STATE.md` ISSUE-S01, `docs/KNOWN_ISSUES.md` ISSUE-001

---

## ADR-008: Manifest as Single Source of Truth for Splits

- **Decision ID**: ADR-008
- **Date**: 2026-07 (inferred)
- **Status**: Accepted
- **Context**: Multiple potential sources of split information existed: config.py split_seed, inline hashing, and the manifest CSV.
- **Decision**: `metadata/dataset_manifest.csv` with SHA-256 checksum is the **single source of truth**. All code reads splits from the manifest via `split_utils.load_manifest_split()`. The manifest is integrity-verified on load.
- **Alternatives considered**: 
  - Dynamic hash-based split (rejected — fragile, hard to audit)
  - Config-driven split (rejected — easy to change accidentally)
- **Rationale**: A checksummed, version-controlled manifest provides immutability and auditability.
- **Implementation effect**: `data/dataset.py` imports from `split_utils`. `load_manifest_split()` validates machine-ID isolation, checks the SHA-256 checksum, and rejects unknown splits.
- **Source evidence**: `utils/split_utils.py`, `metadata/dataset_manifest.csv`, `metadata/dataset_manifest.sha256`

---

## ADR-009: Validation Prediction Exports Use Normalized Relative Paths as Sample IDs

- **Decision ID**: ADR-009
- **Date**: 2026-07-24
- **Status**: Accepted
- **Context**: The EXP-CHAAD-001 validation audit found that `sample_id` values
  generated from `batch_idx * len(labels)` collided when batch size changed
  around a short final batch. The manifest `file_id` column is also not unique
  within validation, while `relative_path` is unique for all validation rows.
- **Decision**: Prediction exporters must use the dataset-level normalized
  manifest `relative_path` as `sample_id` unless a future manifest adds a
  verified unique `sample_id` column.
- **Alternatives considered**:
  - Batch index plus row position (rejected - changes with batch size and can
    collide)
  - Manifest `file_id` (rejected - not unique within validation)
  - Dataset-global integer index (deferred - stable only if manifest ordering is
    also fixed and carried through every dataset wrapper)
- **Rationale**: Normalized relative paths are stable across batch size,
  DataLoader worker count, shuffle order, device, and repeated evaluation.
- **Implementation effect**: `data/dataset.py` exposes `sample_id`,
  `relative_path`, `file_path`, and `split`; `scripts/audit_evaluation_pipeline.py`
  exports and validates those identifiers before saving predictions.
- **Source evidence**: `artifacts/EXP-CHAAD-001/prediction_export_validation.json`,
  `artifacts/EXP-CHAAD-001/prediction_export_determinism.json`,
  `tests/test_prediction_export.py`

---

## ADR-010: Separate API Planner from Sandboxed Local Codex Execution

- **Decision ID**: ADR-010
- **Date**: 2026-07-27
- **Status**: Accepted
- **Context**: A browser ChatGPT conversation cannot directly and continuously
  control a local Codex IDE/CLI session. The project needs resumable,
  evidence-preserving task selection without granting an unattended agent
  authority over protected data or publication actions.
- **Decision**: Use `orchestrator/controller.py` to call an OpenAI Responses API
  planning agent for exactly one structured decision at a time, then invoke
  local `codex exec --json --ephemeral --sandbox workspace-write` only for
  accepted `CODEX_TASK` decisions. Stop on `HUMAN_APPROVAL`, `BLOCKED`, or
  `COMPLETE`.
- **Safety constraints**: Refuse a dirty tree by default; strip common
  credential environment variables from the Codex child; require explicit
  risk flags; require dual protected-test approval; gate scientific experiments
  on research re-entry; use iteration/time limits; persist state atomically;
  require and archive a structured completion report.
- **Alternatives considered**:
  - Copy prompts manually between browser ChatGPT and Codex (rejected as
    fragile and not resumable).
  - Give one unattended prompt authority through submission (rejected because
    it cannot safely represent human, legal, data, or scientific approvals).
  - Pass the planner API key into `codex exec` (rejected because
    repository-controlled commands could inherit it).
- **Implementation effect**: `orchestrator/`, controller-specific requirements,
  ignored runtime/state paths, `reports/autonomous_loop/`, and focused offline
  tests.
- **Verification boundary**: Controller logic, preflight, and dry-run are
  verified locally. No planner API request or nested Codex task has been
  executed.

---

*All decisions inferred from repository evidence. Where exact dates are unknown, approximate dates are marked. See `AGENTS.md` for evidence hierarchy rules.*
