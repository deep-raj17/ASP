# Current Handoff — CHAAD Project

> Quick context for anyone resuming work. Full details in `docs/CURRENT_STATE.md`.

<!-- AUTO-GENERATED-CONTEXT:START -->
| Field | Value |
|-------|-------|
| Date | 2026-07-21 |
| Branch | `blackboxai/research-integrity-audit` |
| Working tree | 5 unstaged modifications, ~7 untracked files |
<!-- AUTO-GENERATED-CONTEXT:END -->

## What Was Most Recently Done

1. **AI project memory system created** — AGENTS.md + 20+ docs/ files
2. **Research integrity audit verified** — 7/7 gates PASS, no data leakage
3. **Shortcut learning audit completed** — No shortcuts detected (metadata AUC=0.59)
4. **Publication pipeline infrastructure built** — 6 new scripts (baselines, statistics, shortcuts, audit, seed, reliability)
5. **Novel contribution implemented** — `models/reliability.py` (reliability-aware fusion)
6. **Reproducibility hardened** — Seed control, provenance tracking added to config/train

## What Is Currently Being Done

**AI project memory system finalization** — Creating remaining docs, ChatGPT context pack, and maintenance script.

## What Remains Unfinished

1. **Model training** — BLOCKED by dataset access (needs MIMII at `E:\MIMII`)
2. **Baseline comparisons** — Script exists, needs checkpoint
3. **Statistical validation** — Script exists, needs predictions
4. **Ablation studies** — Plan exists, needs runner script
5. **Robustness analysis** — Plan exists, needs implementation
6. **Publication manuscript** — Needs all evaluation results

## What Is Blocked

| Item | Blocker |
|------|---------|
| Model training | Dataset at `E:\MIMII` unconfirmed |
| All evaluation scripts | Need `checkpoints/best_model.pt` |
| Publication audit GO verdict | Needs 5 NOT_CHECKED gates resolved |

## What Should Happen Next

1. Verify `E:\MIMII` exists and contains the MIMII dataset
2. Run: `python verify_dataset.py`
3. Run: `python train.py`
4. Run: `python calibrate.py`
5. Run: `python evaluate.py --split test`
6. Run: `python scripts/run_baselines.py`

## What Must Not Be Changed

- Split protocol (machine-independent, manifest-based)
- Threshold selection protocol (validation-only)
- Calibration protocol (train_normal only)
- `metadata/dataset_manifest.csv` and its checksum
- Legacy artifacts in `checkpoints/` and `artifacts/pre_validation_backup/` (historical value)

## Most Relevant Files

| File | Why |
|------|-----|
| `config.py` | Edit dataset_dir first |
| `train.py` | Training entry point (runs provenance + seed) |
| `models/reliability.py` | Novel contribution |
| `metadata/dataset_manifest.csv` | Split truth |
| `_audit_check.py` | Integrity verification |
| `scripts/run_publication_audit.py` | Go/no-go status |
| `docs/CURRENT_STATE.md` | Full capability inventory |

## Warning: Do Not Trust Legacy Metrics

`checkpoints/eval_report.json` shows ROC-AUC 99.99997% — this was on the **validation split** under an **unknown split protocol**. These values are LEGACY and cannot be cited. All official metrics must come from `python evaluate.py --split test` with the current manifest.

---

*Handoff expires: next significant code change | Update `docs/CURRENT_STATE.md` after each session*
