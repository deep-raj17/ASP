# Current State — CHAAD Project

> **Last updated:** 2026-07-27

<!-- AUTO-GENERATED-CONTEXT:START -->
| Field | Value |
|-------|-------|
| Branch | `blackboxai/research-integrity-audit` |
| Working tree | DIRTY; preservation verification began with 69 staged entries, 13 entries with unstaged changes, and 38 untracked entries |
<!-- AUTO-GENERATED-CONTEXT:END -->

## Current Project Phase

**Research Validation & Publication Preparation** — The first
machine-independent training run is complete and preserved as `EXP-CHAAD-001`.
The validation prediction-export bug has been fixed and Prompt 2 passes; Prompt
3 still indicates model underfitting. The ROS-PROJECT-01 adapter imported the
state without rewriting historical evidence. A new full-corpus audit verified
all 53,046 WAV files for readability, finite samples, metadata, and current
SHA-256 with zero errors. PMPS-01 is now machine-derived **BLOCKED**, not
FAILED: 12 of 13 requirements are satisfied, while the extracted local corpus
still lacks cryptographic acquisition evidence tying it to the official MIMII
public 1.0 archives. Later PMPS gates are not authorized.

## Production Inference Readiness

**BLOCKED** — `checkpoints/best_model.pt` exists, but
`checkpoints/detector_calibration.pt` is missing. The UI, Flask API, and
inference verifier now fail closed instead of serving uncalibrated or
random-weight output. Model and calibration artifact loading uses restricted
PyTorch deserialization. This behavior is VERIFIED by three focused tests and
the full 73-test suite on 2026-07-27.

## Local Autonomous Orchestration

**IMPLEMENTED AND OFFLINE-VERIFIED** — `orchestrator/controller.py` can connect
an OpenAI Responses API planning agent to local non-interactive `codex exec`
tasks. The controller has preflight and dry-run modes that make no API or Codex
task call, refuses dirty-worktree execution unless explicitly overridden,
removes common credential variables from the Codex child environment, validates
planner decisions, stops at declared human-risk gates, and requires an updated
completion report after each successful iteration.

Protected-test access remains unauthorized and requires both the controller
state flag and a scoped approval record. Scientific experiments remain gated
by explicit research re-entry. The controller has not performed an API-backed
iteration; only its focused offline tests, preflight, and dry run have been
executed.

## Current Objective

ROS-PUB-01 was evaluated on 2026-07-24 and is **BLOCKED** by its prerequisite
rule: ROS-PROJECT-02 through ROS-PROJECT-13 have not been completed, and the
imported PMPS workflow remains PMPS-01 **BLOCKED**. Contribution discovery,
novelty, falsification, and manuscript-readiness outputs have intentionally not
been generated. ROS-PUB-02 is not authorized until ROS-PUB-01 is PASS.

Complete the publication pipeline: run baselines, statistical validation, ablation studies, and robustness analysis after training a model. Produce a conference submission.

## Capability Inventory

ROS-PUB-02 and ROS-PUB-03 were evaluated on 2026-07-24 and are **BLOCKED** by
their prerequisite rules. Venue selection and manuscript architecture outputs
were not generated.

ROS-PUB-04 was evaluated on 2026-07-24 and is **BLOCKED** because ROS-PUB-03
did not pass. Independent scientific audit and adversarial peer-review outputs
were not generated.

ROS-PUB-05 is also **BLOCKED** because all four preceding publication gates
must pass before submission-package assembly can begin.

ROS-DEPLOY-01 and ROS-SEC-01 are **BLOCKED** by incomplete upstream project and
publication domains. Deployment, security, privacy, and trust assessments have
not been started.

ROS-DATA-01 is **BLOCKED** because ROS-SEC-01 and the other upstream domains
are incomplete. Data-governance and lifecycle assessments have not started.

ROS-DATA-01 Part 2 was checked and remains blocked; no partial data assessment
is considered complete.

ROS-DATA-01 Part 3 was checked and remains blocked by the existing upstream
gate state.

ROS-DATA-01 Part 4 remains blocked; the data-governance lifecycle has no
completed certificate.

ROS-SEC-01 Parts 2 and 3 were checked and remain blocked under the same
upstream prerequisite failure; no partial security assessment is considered
complete.

### A. VERIFIED AND WORKING

| Capability | Evidence | Last Validated |
|------------|----------|-----------------|
| Data split protocol (machine-independent, 3-split) | `_audit_check.py` output: train=12,045, val=28,254, test=12,747 | 2026-07-21 |
| Machine-ID isolation (no cross-split leakage) | `artifacts/research_audit/machine_split_table.csv` | 2026-07-21 |
| SHA-256 duplicate detection (0 cross-split) | `artifacts/research_audit/duplicate_hash_report.csv` | 2026-07-21 |
| Fixed-constant normalization (no data leakage) | `utils/audio_utils.py` lines 93-97 | 2026-07-21 |
| Calibration on train_normal only | `calibrate.py`, `inference/detector.py` | 2026-07-21 |
| Threshold selection via Youden's J on validation | `utils/metrics.py` lines 97-100 | 2026-07-21 |
| Continuous-score metric computation (sklearn) | `utils/metrics.py` | 2026-07-21 |
| Shortcut learning audit (metadata AUC=0.59) | `scripts/audit_shortcuts.py` executed | 2026-07-21 |
| Deterministic seed control | `utils/seed.py` | 2026-07-21 |
| Dataset manifest content | `metadata/dataset_manifest.csv`; calculated SHA-256 recorded in EXP-CHAAD-001 | 2026-07-24 |
| Subgroup distribution analysis | `reports/per_machine_results.csv` etc. | 2026-07-21 |
| EXP-CHAAD-001 checkpoint preservation | Source/copy SHA-256 match; epoch 6 metadata and 100-epoch TensorBoard history inspected | 2026-07-24 |

### B. IMPLEMENTED BUT UNVERIFIED

| Capability | Code Location | Validation Needed |
|------------|---------------|-------------------|
| HybridAnomalyModel training pipeline | `train.py`, `training/trainer.py`, `artifacts/EXP-CHAAD-001/` | Training audit classified **MODEL UNDERFITTING**; no architecture change or retraining performed |
| Multi-signal anomaly detection | `inference/detector.py` | Verify on real samples after calibration |
| Model evaluation on validation split | `scripts/audit_evaluation_pipeline.py`, `artifacts/EXP-CHAAD-001/evaluation_audit.md` | **VERIFIED for validation export**: corrected CSV has 28,254 rows, 28,254 unique IDs, 0 duplicates; validation ROC-AUC 0.6002609445 |
| Model evaluation on held-out test split | `evaluate.py` | Not run in this diagnostic sequence; test split remains untouched |
| Reliability-aware fusion module | `models/reliability.py` | Train on validation data, test on held-out test |
| Baseline comparisons (11 baselines) | `scripts/run_baselines.py` | Execute with model checkpoint |
| Statistical validation (bootstrap, DeLong, etc.) | `scripts/statistical_validation.py` | Execute with predictions CSV |
| Production detector export | `inference/production_detector.py` | Test end-to-end inference |
| Edge deployment pipeline | `edge_deploy/` | Deploy to Raspberry Pi and test |
| Data integrity tests | `tests/test_data_integrity.py` | Execute and verify results |

### C. PARTIALLY IMPLEMENTED

| Capability | Status | Remaining |
|------------|--------|-----------|
| Ablation studies | Ablation plan in `docs/NOVELTY_AND_CONTRIBUTIONS.md` Section H | Need to implement runner script and execute |
| Robustness analysis | `docs/LIMITATIONS.md` exists | Need perturbation tests, confusion analysis |
| Publication-quality evaluation report | Infrastructure scripts created | Need model predictions to populate |

### D. IN PROGRESS

| Item | Files Modified | Status |
|------|---------------|--------|
| AI project memory system | `AGENTS.md`, `docs/AI_CONTEXT_INDEX.md`, multiple doc files being created | Active (2026-07-21) |
| Reproducibility hardening | `config.py` (+random_seed), `train.py` (+provenance), `utils/seed.py` | Unstaged changes |

### E. PLANNED

See `docs/ROADMAP.md` for full details. Summary:

1. Independently audit the preserved run before any architecture changes
2. Run baseline comparisons
3. Run statistical validation
4. Implement and run ablation studies
5. Implement robustness analysis
6. Generate publication manuscript
7. Create camera-ready submission package

### F. FAILED / ABANDONED / SUPERSEDED

| Item | Fate | Evidence |
|------|------|----------|
| File-hash-based split (old protocol) | SUPERSEDED by manifest-based split | `data/dataset.py` now uses `split_utils.load_manifest_split()` |
| `_inspect_manifest.py` | DELETED (temporary diagnostic) | Removed during Phase 4 cleanup per `TODO.md` |
| Stale `machine_dependent` config | SUPERSEDED by `machine_independent` | `_audit_check.py` uses manifest directly |

### G. BLOCKED

| Item | Blocker | Resolution |
|------|---------|------------|
| Statistical validation | Requires predictions CSV from model run | Run baselines/evaluation first |
| Edge deployment testing | Requires Raspberry Pi hardware | Physical device needed |

## Current Architecture Summary

The system follows a pipeline: Dataset → Manifest-based Split → Audio Preprocessing → Training (CNN+Transformer+AE+Contrastive) → Calibration (train_normal) → Evaluation (val threshold, test metrics) → Deployment (ONNX/edge export).

The novel contribution is `models/reliability.py` — a reliability-aware fusion module that learns sample-dependent weights for combining four anomaly signals. This sits after calibration and before the final threshold.

## Current Testing Status

- **Unit tests:** `tests/test_data_integrity.py` — UNVERIFIED (not executed in this session)
- **Audit scripts:** `_audit_check.py` — VERIFIED (7/7 gates pass)
- **Shortcut audit:** `scripts/audit_shortcuts.py` — VERIFIED (no shortcuts detected)
- **Publication audit:** `scripts/run_publication_audit.py` — VERIFIED (57.1%, CONDITIONAL)

## Active Known Errors

- EXP-CHAAD-001 has low validation discrimination after corrected export (ROC-AUC 0.6002609445) and Prompt 3 classified the current training behavior as **MODEL UNDERFITTING**.
- The selected checkpoint is epoch 6 by minimum validation loss, not epoch 100 or the maximum-AUC epoch 11.
- The original `artifacts/EXP-CHAAD-001/validation_predictions.csv` is a corrupted historical export with 30 duplicated sample IDs (60 affected rows); the corrected artifact is `validation_predictions_corrected.csv`.
- **RESOLVED BY CURRENT FILE EVIDENCE:** `metadata/dataset_manifest.sha256`
  currently matches the manifest bytes exactly:
  `7c689508cbed4d49d05ec2891b315b27722ff01a8a62b6b1c4f610e3afcd0136`.
  Earlier preservation notes captured a mismatch; the current matching bytes
  supersede that stale state, but the change origin remains unproven.
- Minor: `scripts/audit_shortcuts.py` produces a scipy RuntimeWarning about precision loss in duration comparison (harmless).

## Immediate Next Action

1. Obtain the original MIMII ZIP archives, a trustworthy download receipt, or
   authoritative per-file checksums to prove local release identity. The
   official record identifies public 1.0 and CC BY-SA 4.0, but folder agreement
   alone is not cryptographic identity evidence.
2. Register and verify that provenance, then machine-re-evaluate PMPS-01.
3. Do not begin ROS-PROJECT-02 or later PMPS gates while this critical blocker
   remains.
4. Do not use the test split, retrain, or alter architecture during this audit.

## Definition of Next Completion Milestone

**Publication audit score ≥ 90%** — all 10 gates passing or at minimum conditionally passing with documented rationale. This requires: trained model, baseline results, statistical validation, ablation study report, and robustness analysis.

---

*Evidence table format: | Item | Status | Evidence | Last validated | Next action |*
