# ROS-CLI-01 Test Report

`python -m pytest tests\ros\e2e\test_cli.py -q` — **PASS, 10 tests**.

Isolated temporary-workspace coverage includes help/version, every top-level
command name, valid and invalid commands, project add/show/list/history/
validation, text and stable JSON output, correlation IDs, init and workflow
dry-run, non-interactive import preview, registry show/history/list/rebuild/
verify/export, doctor, idempotent replay and conflicting replay, archive
approval, unsafe output rejection, explicit deferred-command failures, absence
of manual gate pass, secret masking, and interrupted-command exit 130.

Type checking, Ruff linting, and Black formatting are **NOT RUN** because those
tools are not installed in the existing environment. No unavailable check is
reported as passing.
