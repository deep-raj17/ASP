# Workflow Engine Implementation

`WorkflowEngine` is a deterministic service over a validated definition,
`JsonStateStore`, and `AppendOnlyAuditLog`. It exposes instance creation,
transition requests, explicit gate evaluation, and read-only planning. It never
executes scientific tasks or verifies evidence.
