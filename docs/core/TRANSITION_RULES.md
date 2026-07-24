# Transition Rules

Commands carry actor, correlation/idempotency keys, expected revision, reason,
and dry-run. State changes require an allowed edge and satisfied prerequisites.
Cancellation requires justification. Waiver requires both policy and approval.
Repeated identical idempotency keys return the original result.
