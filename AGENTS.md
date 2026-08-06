# AI Agent Instructions — CHAAD Project

> This file governs how AI coding agents (ChatGPT, Claude, Copilot, etc.) must interact with this repository. All agents MUST read this file before making any changes.

## Mandatory Reading Order

Before editing any code, read these files in order:

1. **AGENTS.md** (this file)
2. **docs/AI_CONTEXT_INDEX.md** — master index and project summary
3. **docs/PROJECT_CONTEXT.md** — what, why, who, and how
4. **docs/CURRENT_STATE.md** — what is true right now, verified vs. unverified
5. **docs/ARCHITECTURE.md** — system design, data flow, component map
6. **docs/DECISIONS.md** — key technical and research decisions
7. **docs/ROADMAP.md** — approved future work, milestones, priorities
8. **docs/KNOWN_ISSUES.md** — active bugs, blockers, and limitations

For ML-specific work, also read:

9. **docs/DATASET_AND_SPLITS.md** — dataset specification and split protocol
10. **docs/DATA_LEAKAGE_AUDIT.md** — leakage checks and verification status
11. **docs/REPRODUCIBILITY.md** — seed control, environment, determinism

## Evidence Hierarchy

When documentation conflicts with other sources, prefer (in order):

1. **Source code** — the definitive implementation
2. **Executed test / script output** — verifiable artifacts
3. **Git history** — commits, diffs, timestamps
4. **Generated reports** — `artifacts/`, `reports/`, `checkpoints/`
5. **Configuration files** — `config.py`, `configs/`
6. **Documentation** — `docs/`, `README.md` (least authoritative)

When you find a conflict, **report it explicitly** rather than silently choosing one interpretation.

## Rules for Making Changes

### Before Changes
- Understand the request fully
- Read relevant context documents (above)
- Inspect the current implementation by reading source files
- Check `git status` for uncommitted work
- Identify all files that will be affected
- State your implementation plan explicitly
- Identify risks and assumptions

### During Changes
- Make **one focused change at a time**
- Do NOT batch unrelated modifications in a single commit
- Preserve existing functionality — do not refactor unless explicitly requested
- Use existing patterns, naming conventions, and module structure
- Run validation after each change

### After Changes
- Run relevant tests or validation scripts
- List every file modified
- Explain what behaviour changed and why
- Report test/validation results (pass or fail)
- Update `docs/CURRENT_STATE.md` if the project state changed
- Update `docs/CHANGELOG.md` with your change
- Update `docs/SESSION_LOG.md` with a session entry
- If a significant technical decision was made, add to `docs/DECISIONS.md`
- If an experiment ran, add to `docs/EXPERIMENT_LOG.md`

### Autonomous Controller Tasks

When a task is launched through `orchestrator/controller.py`, Codex must also
write `reports/autonomous_loop/LATEST_CODEX_REPORT.md` with:

1. Objective
2. Files inspected
3. Files changed
4. Commands executed
5. Tests executed
6. Results
7. Failures
8. Evidence paths
9. Acceptance-criteria assessment
10. Remaining risks
11. Recommended next action

The autonomous controller does not broaden permissions in this file. Dataset
downloads or writes, scientific experiments, protected-test access, external
writes, destructive actions, paid resources, Git push/merge, and manuscript
submission still require their applicable explicit human authorization.

### Do NOT
- Fabricate test results, metrics, or completion claims
- Mark planned features as implemented
- Treat old documentation as current without verification
- Delete user files or data without explicit authorization
- Expose secrets, API keys, or private paths
- Commit or push code automatically
- Deploy to production without explicit authorization
- Install packages without asking
- Download datasets without asking
- Retrain models without asking
- Write an `escapeHtml` or HTML-escaping utility function (runtime handles this)

## Verification Language

Use precise language when reporting status:

| Term | Meaning |
|------|---------|
| VERIFIED | Confirmed by executing code and observing output |
| PARTIALLY VERIFIED | Some but not all checks passed |
| UNVERIFIED | Code exists but has not been tested |
| IN PROGRESS | Currently being modified |
| PLANNED | Appears in roadmap but not yet implemented |
| PROPOSED | Discussed but not approved |
| FAILED | Attempted, did not work, has evidence of failure |
| ABANDONED | Started but intentionally discontinued |
| SUPERSEDED | Replaced by a newer approach |
| BLOCKED | Cannot proceed due to dependency or uncertainty |
| UNKNOWN | Cannot determine from available evidence |

Never use "all tests pass" unless you actually ran all tests and they all passed.

## Project-Specific Rules

- Dataset path: `E:\MIMII` (configurable in `config.py` line 14)
- Split protocol: **Machine-independent** (each machine ID in exactly one split)
- Splits: train=id_04 (12,045), val=id_00+id_02 (28,254), test=id_06 (12,747)
- The manifest (`metadata/dataset_manifest.csv`) is the **single source of truth** for splits
- `_audit_check.py` dynamically computes all decision gates from the manifest
- On Windows: `num_workers` must be 0 to avoid multiprocessing issues
- The novel contribution is the **reliability-aware fusion module** (`models/reliability.py`)

## Quick Start Commands

```bash
# Verify dataset structure
python verify_dataset.py

# Run research integrity audit
python _audit_check.py

# Run shortcut learning audit
python scripts/audit_shortcuts.py

# Run publication go/no-go audit
python scripts/run_publication_audit.py --verbose

# Train (requires MIMII dataset at configured path)
python train.py

# Calibrate detector on normal training data
python calibrate.py

# Evaluate on validation or test split
python evaluate.py --split test
```

---

*Last updated: 2026-07-21 | Commit: see `git rev-parse HEAD`*
