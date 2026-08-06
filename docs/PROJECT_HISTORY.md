# Project History — CHAAD

> Reconstructed from Git history, file timestamps, documentation, and source code evidence. Entries without commit hashes are approximate/inferred.

## Chronological Development

### Phase 0: Foundation (pre-2025)
- **Status**: UNKNOWN — repository was initialized before the current audit branch
- Initial implementation: HybridAnomalyModel architecture (EfficientNet-B4 + Transformer + Autoencoder)
- Training pipeline, multi-objective loss, calibration protocol
- Dataset manifest generation and split utilities

### Phase 1: Research Infrastructure (2025 – early 2026)
| Event | Evidence | Status |
|-------|----------|--------|
| Model training and evaluation pipeline | `train.py`, `evaluate.py`, `training/trainer.py` | VERIFIED (code exists) |
| Multi-signal anomaly detector | `inference/detector.py` | VERIFIED (code exists) |
| Calibration on train_normal | `calibrate.py` | VERIFIED |
| Edge deployment (Raspberry Pi) | `edge_deploy/` | IMPLEMENTED but UNVERIFIED |
| Production API | `production_api.py`, `PRODUCTION.md` | IMPLEMENTED |
| Docker containerization | `Dockerfile`, `docker-compose.yml` | IMPLEMENTED |

### Phase 2: Research Integrity Audit (2026-07)
| Event | Evidence | Status |
|-------|----------|--------|
| Branch created: `blackboxai/research-integrity-audit` | Git branch | VERIFIED |
| Dataset manifest generated | `metadata/dataset_manifest.csv` (53,046 samples) | VERIFIED |
| Split protocol formalized (machine-independent, 3-split) | `utils/split_utils.py`, `_audit_check.py` | VERIFIED |
| Machine-ID isolation verified | `artifacts/research_audit/machine_split_table.csv` | VERIFIED |
| SHA-256 duplicate check (0 cross-split) | `artifacts/research_audit/duplicate_hash_report.csv` | VERIFIED |
| Normalization audit (fixed constants) | `artifacts/research_audit/normalization_audit.json` | VERIFIED |
| Calibration audit (train_normal only) | `artifacts/research_audit/calibration_audit.json` | VERIFIED |
| Threshold audit (Youden's J on val) | `artifacts/research_audit/threshold_audit.json` | VERIFIED |
| Old file-hash split SUPERSEDED by manifest split | `data/dataset.py` | VERIFIED |
| `_audit_check.py` executed — 7/7 gates PASS | Terminal output | VERIFIED |

### Phase 3: Publication Preparation (2026-07-21)
| Event | Evidence | Status |
|-------|----------|--------|
| Novelty document finalized | `docs/NOVELTY_AND_CONTRIBUTIONS.md` | VERIFIED |
| Reliability-aware fusion designed | `models/reliability.py` (created) | IMPLEMENTED, UNVERIFIED |
| Baseline comparison script | `scripts/run_baselines.py` (11 baselines) | IMPLEMENTED |
| Statistical validation script | `scripts/statistical_validation.py` | IMPLEMENTED |
| Shortcut learning audit | `scripts/audit_shortcuts.py` — executed, PASS | VERIFIED |
| Publication go/no-go audit | `scripts/run_publication_audit.py` — 57.1% | VERIFIED |
| Deterministic seed control | `utils/seed.py` | IMPLEMENTED |
| Reproducibility hardening | `config.py` (+seeds), `train.py` (+provenance) | UNSTAGED |
| AI project memory system | `AGENTS.md`, `docs/` files | IN PROGRESS |

## Decision Timeline (Inferred)

1. **Backbone selection**: EfficientNet-B4 chosen over ResNet-50 for better mel-spectrogram feature extraction (config default)
2. **Temporal module**: Transformer preferred over BiLSTM (config default)
3. **Fusion strategy**: Hand-selected weights (0.30/0.25/0.30/0.15) in `InferenceConfig` were the original approach
4. **Split protocol change**: Original file-hash-based split (15% val) was **superseded** by manifest-based machine-independent 3-split
5. **Novelty pivot**: The project evolved from "engineering integration of known methods" to having a clear novel contribution (reliability-aware fusion), as documented in `docs/NOVELTY_AND_CONTRIBUTIONS.md`
6. **Audit-driven**: The research integrity audit (`_audit_check.py`) became the gatekeeper for all experimental claims

## Key Artifacts

| Artifact | Date | Description |
|----------|------|-------------|
| `metadata/dataset_manifest.csv` | Pre-audit | 53,046 WAV files with splits, SHA-256, metadata |
| `checkpoints/calibration_report.json` | 2026-05-15 | Reference pool: 37,685 normal samples |
| `checkpoints/eval_report.json` | Unknown | Reported ROC-AUC: 0.9999997 (VALIDATION SET — needs independent test verification) |
| `artifacts/pre_validation_backup/` | Pre-audit | Backup of configs before validation protocol changes |

## Unresolved Historical Questions

1. **When was the model actually trained?** The checkpoint timestamps suggest May 2026, but no training provenance was recorded before the audit.
2. **What was the original reported ROC-AUC?** The `checkpoints/eval_report.json` shows 99.99997% — this was evaluated on the VALIDATION set, not a held-out test set. The current audit protocol requires independent test-set evaluation.
3. **Were any experiments run with the old file-hash split?** The stale `machine_dependent` config in old reports suggests yes, but those results have been superseded.

---

*Evidence quality: Git history is limited to the current branch. Exact dates before the audit are approximate.*
