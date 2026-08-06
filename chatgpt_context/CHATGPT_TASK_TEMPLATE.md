# Task Template — CHAAD Project

> Use this template when requesting work from ChatGPT or any AI agent.

---

# Task: [Brief title]

## Objective

[One paragraph describing what should be accomplished and why.]

## Reason

[Why this task matters to the project. Link to roadmap milestone or issue ID.]

## Expected Result

[What should exist or be true when the task is complete.]

## Relevant Files

- `path/to/file.py` — [why this file matters]
- `path/to/another.py` — [why]

## Constraints

- Must not change split protocol
- Must not modify `metadata/dataset_manifest.csv`
- Must preserve existing functionality in [module]
- [Any other constraints]

## Must Preserve

- [Capability or file that must remain unchanged]
- [Protocol or convention that must be followed]

## Validation Requirements

- [ ] Run `python _audit_check.py` and verify 7/7 gates PASS
- [ ] Run `python scripts/run_publication_audit.py` and verify no regressions
- [ ] [Test command] should produce [expected output]
- [ ] Type checking: `python -c "import py_compile; ..."` should succeed

## Documentation Updates

- [ ] Update `docs/SESSION_LOG.md` with session entry
- [ ] Update `docs/CURRENT_STATE.md` if state changed
- [ ] Update `docs/ROADMAP.md` if milestone completed
- [ ] Update `docs/DECISIONS.md` if a decision was made
- [ ] Update `docs/EXPERIMENT_LOG.md` if an experiment ran

## Definition of Done

- [ ] All validation requirements pass
- [ ] All documentation updates complete
- [ ] No new blockers introduced
- [ ] Git status shows only intended changes
- [ ] Session log entry written with suggested commit message
