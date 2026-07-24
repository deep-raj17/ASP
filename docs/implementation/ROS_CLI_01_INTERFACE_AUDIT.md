# ROS-CLI-01 Interface Audit

Stable service facades exist for workflows and registries. CLI handlers call
`RosServices`; they do not open or edit storage directly. No command exposes
destructive deletion or manual scientific gate satisfaction.

The implemented surface covers initialization, project registration and
inspection, status, registry/project/workflow verification, workflow
start/planning, gate requirements, registry inspection/import/export/rebuild,
doctor, and approval-protected archive. Commands that require a future
evidence/module service facade are accepted by the parser but fail explicitly
with `COMMAND_NOT_IMPLEMENTED`; they never return a misleading success result.

All mutating implemented operations accept global `--dry-run`. JSON responses
use a versioned envelope. Output-path confinement, approval enforcement,
secret redaction, interruption handling, and stable exit mappings are exercised
by isolated end-to-end tests.
