# ROS-CORE-03 Test Report

## Executed

`python -m pytest tests\ros\unit\test_registry.py -q` — **PASS, 8 tests**.

Coverage includes append, duplicate/idempotent append, exact versions, history,
supersession, deprecation, tombstones, current views, rebuilds, cross-registry
references, project/workflow/experiment queries, payload and metadata tamper
detection, validated atomic import/export, lineage import, restart recovery,
dry-run behavior, optimistic concurrency, FAILED experiment preservation, and
INCOMPLETE record visibility.

The integration scenario registers a project, workflow, evidence artifact,
FAILED experiment, model, and publication with cross-registry lineage, then
reopens the database and verifies both integrity and record visibility.

## Quality checks

- Python compilation: PASS as part of the final ROS verification.
- Type checking: NOT RUN; mypy is not installed in the existing environment.
- Linting: NOT RUN; Ruff is not installed in the existing environment.
- Formatting verification: NOT RUN; Black is not installed in the existing
  environment.

No unavailable check is reported as passing.
