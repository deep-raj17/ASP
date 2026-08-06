# ChatGPT Start Prompt — CHAAD Project

> Copy and paste this prompt at the start of a new ChatGPT conversation about this project.

---

Read the attached project context documents before answering or making any changes.

First, reconstruct the project state by reading these files in order:

1. `AGENTS.md`
2. `docs/AI_CONTEXT_INDEX.md`
3. `docs/PROJECT_CONTEXT.md`
4. `docs/CURRENT_STATE.md`
5. `docs/ROADMAP.md`
6. `docs/KNOWN_ISSUES.md`

For ML-specific work, also read:

7. `docs/DATASET_AND_SPLITS.md`
8. `docs/DATA_LEAKAGE_AUDIT.md`
9. `docs/REPRODUCIBILITY.md`

Then, before proposing any changes, explicitly distinguish:

- **VERIFIED completed work** (executed and confirmed)
- **IMPLEMENTED BUT UNVERIFIED work** (code exists, not tested)
- **WORK IN PROGRESS** (actively being modified)
- **PLANNED work** (on the roadmap, not started)
- **PROPOSED but unapproved ideas** (discussed, no approval)
- **FAILED, ABANDONED, or SUPERSEDED work** (do not revive)
- **BLOCKED work** (cannot proceed due to dependency)

Use source code, executed test/script output, Git evidence, and reproducible artifacts as stronger evidence than documentation or comments.

Do NOT make changes yet. First report:

1. Your understanding of the project
2. The current phase and objective
3. The most recently completed work
4. Active work in progress
5. Active blockers
6. Current validation status (which audits pass, which are unchecked)
7. The approved next action per the roadmap
8. Any conflicting information you found between documentation and source code

After I confirm your understanding, we can proceed with the specific task.

---

**Critical warnings to remember throughout the conversation:**

- Legacy metrics (ROC-AUC 99.99997% in `checkpoints/eval_report.json`) are from an unknown split protocol and are NOT valid final results
- Model checkpoint may not exist — verify before assuming
- Dataset at `E:\MIMII` may not be accessible — verify before training
- The novel contribution is `models/reliability.py` — do not claim the base architecture is novel
- All official metrics must come from `python evaluate.py --split test`
- Never fabricate test results, metrics, or completion claims
