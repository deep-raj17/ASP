# AI Context Index — CHAAD Project

> **This is the starting point for any AI agent entering this project.** Read this file first.

## Project Identity

| Field | Value |
|-------|-------|
| **Name** | CHAAD (Configurable Hybrid Acoustic Anomaly Detection) |
| **Type** | AI/ML Research — Supervised industrial acoustic anomaly detection |
| **Domain** | Factory sound monitoring (fans, pumps, sliders, valves) |
| **Dataset** | MIMII (Malfunctioning Industrial Machine Investigation and Inspection) |
| **Publication Target** | IEEE/CVPR/ICASSP conference paper |
| **Current Phase** | Research validation and publication preparation |
| **Git Branch** | `blackboxai/research-integrity-audit` |

## Executive Summary

CHAAD is a hybrid deep-learning system that detects anomalous sounds from industrial machines. It combines a CNN backbone (EfficientNet-B4), Transformer temporal module, attention pooling, autoencoder branch, and contrastive learning into a single multi-signal anomaly detector. The novel research contribution is a **reliability-aware fusion module** (`models/reliability.py`) that learns sample-dependent weights for combining four anomaly signals (reconstruction, embedding distance, Mahalanobis distance, contrastive distance) conditioned on machine type and noise condition.

The project has undergone a rigorous research integrity audit. All data leakage checks pass. A three-split protocol with machine-independent isolation is verified. The current publication audit scores 57.1% (5/10 gates pass) — no critical failures, but 5 gates require a trained model checkpoint to execute.

## Current Objective

Complete remaining publication gates (baselines, statistical validation, ablations, robustness analysis) to reach GO verdict (>90%), then generate manuscript for conference submission.

## Most Important Next Action

Complete the PMPS-01 full-corpus integrity gate: live-decode all 53,046 WAV
files, scan decoded samples for NaN/Inf, and recompute current-file SHA-256
values before starting PMPS-02A.

## Active Blockers

- **PMPS-01 full-corpus verification incomplete**: manifest/file existence and
  size checks pass, but only one representative WAV was live-decoded in the
  current validation run. Full finite-value and live-hash verification remain
  unverified.
- **Dataset identity incomplete**: the local MIMII version and redistribution
  license are unknown.
- **Cross-platform reproducibility unverified**: the current evidence is from
  one Windows/CUDA environment.

The earlier checkpoint blocker is resolved: `checkpoints/best_model.pt` exists
and its verified SHA-256 is
`7d58293ea5138b730811bb261de0b971a8a5e4e52665961b48f2f4a4d85ad7e9`.

## Last Known Validation State

| Check | Status | Date |
|-------|--------|------|
| Research Integrity Audit (`_audit_check.py`) | 7/7 gates PASS | 2026-07-21 |
| Shortcut Learning Audit | PASS (metadata AUC=0.59) | 2026-07-21 |
| Publication Go/No-Go Audit | 57.1% (5/10 PASS) | 2026-07-21 |

## Last Documented Git Commit

Branch: `blackboxai/research-integrity-audit`. Staged: 62 files. Unstaged modifications: 5 files. Untracked: 7 files.

## Mandatory Reading Order

1. **AGENTS.md** — agent rules and project conventions
2. **docs/AI_CONTEXT_INDEX.md** ← you are here
3. **docs/PROJECT_CONTEXT.md** — full project description, motivation, tech stack
4. **docs/CURRENT_STATE.md** — verified vs. unverified capability inventory
5. **docs/ARCHITECTURE.md** — system design, data flow (already exists, may need updating)
6. **docs/DECISIONS.md** — key technical and research decisions
7. **docs/ROADMAP.md** — approved future work
8. **docs/KNOWN_ISSUES.md** — bugs and blockers
9. **docs/DATASET_AND_SPLITS.md** — MIMII dataset and split protocol
10. **docs/DATA_LEAKAGE_AUDIT.md** — leakage verification
11. **docs/REPRODUCIBILITY.md** — seed control and environment

## Document Index

| Document | Purpose | Status |
|----------|---------|--------|
| `PROJECT_CONTEXT.md` | Full project description, problem, users, tech stack | New |
| `CURRENT_STATE.md` | Verified capability inventory with evidence | New |
| `ARCHITECTURE.md` | System design, component map, data flow | Existing (review needed) |
| `PROJECT_HISTORY.md` | Chronological development history from Git | New |
| `DECISIONS.md` | Architecture Decision Records | New |
| `ROADMAP.md` | Approved future work with priorities | New |
| `CHANGELOG.md` | Human-readable change log | New |
| `KNOWN_ISSUES.md` | Confirmed bugs, blockers, limitations | New |
| `TECHNICAL_DEBT.md` | Code quality and maintenance items | Existing (review needed) |
| `ENVIRONMENT.md` | Setup, dependencies, platform assumptions | New |
| `TESTING_AND_VALIDATION.md` | Test inventory, commands, current results | New |
| `SESSION_LOG.md` | Reusable session log template | New |
| `DATASET_AND_SPLITS.md` | MIMII dataset specification | New |
| `EXPERIMENT_LOG.md` | Past and current experiment records | New |
| `MODEL_CARD.md` | Model architecture, training, intended use | New |
| `METRICS_REGISTRY.md` | Central metric values with evidence | New |
| `DATA_LEAKAGE_AUDIT.md` | Comprehensive leakage verification | New |
| `REPRODUCIBILITY.md` | Exact reproduction instructions | New |

## Resolving Documentation Conflicts

When documentation conflicts with source code, tests, Git history, or generated artifacts:

1. **Do not silently choose one interpretation.**
2. Report the conflict explicitly.
3. Determine which source is more recent and more strongly verified.
4. Source code and executed output are stronger evidence than documentation.
5. Document the resolution in the relevant context file.

---

*Generated: 2026-07-21 | Project root: c:/ASP/ASP | By: AI project memory system audit*
