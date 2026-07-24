# ROS-CORE-01 Test Report

`python -m pytest tests/ros -q`: 22 passed after two detected issues were fixed.
CORE-01 coverage includes loading, schema rejection, cycles, prerequisites,
evidence references, waiver authorization, dry-run, idempotency, append-only
audit, concurrency, cancellation, blocking, failure, retry, and restart.

Syntax compilation and dependency checks are part of the final verification.
mypy, Ruff, and Black were unavailable; those checks are reported as NOT RUN,
not PASS.
