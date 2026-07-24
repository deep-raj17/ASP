# Automation Services Specification

Specification: ROS-FS-01  
Version: Draft 1.0

Automation services consume authorized task intents and emit artifacts/events.
They do not write authoritative gate verdicts.

## Services

| Service | Responsibility | Inputs | Outputs |
|---|---|---|---|
| Experiment Runner | Execute pinned protocols | experiment, environment, approval | logs, checkpoints, result artifacts |
| Benchmark Engine | Fair comparative executions | benchmark plan, datasets, models | measurements, comparison artifacts |
| Statistics Engine | Predeclared analyses | result evidence, analysis plan | tests, intervals, effect-size artifacts |
| Paper Engine | Assemble evidence-backed drafts | verified claims/tables/figures | manuscript artifacts, traceability |
| Artifact Builder | Package content-addressed bundles | artifact set, policy | manifest, archive, checksums |
| Release Builder | Prepare approved releases | verified bundle, approvals | release candidate and metadata |
| Dashboard Backend | Serve projections | registry read models | status/query responses |
| Event Bus | Transport ordered events | event envelopes | durable consumer delivery |
| Notification System | Deliver non-authoritative notices | events, subscriptions | delivery receipts |

## Execution requirements

Every service declares version, accepted schemas, permissions, resource limits,
idempotency behavior, retry policy, and emitted events. Inputs are pinned by
digest. Outputs are written to unique destinations and ingested only after
digest verification.

## Event bus

Delivery is at least once. Consumers deduplicate by event ID and maintain
per-aggregate sequence. Producers use an atomic outbox with registry append.
Consumers never assume global ordering. Dead-letter events retain payload,
error, attempts, and replay authorization.

## Determinism

Services identify deterministic, seed-controlled, tolerance-deterministic, and
nondeterministic operations. Repeated executions create distinct attempts.
Nondeterminism is measured and reported, never hidden.

## External side effects

Submission, publication, release, notification to people, paid compute, and
destructive operations require policy evaluation and appropriate approval.
Idempotency keys prevent duplicate side effects. Reconciliation records
external identifiers and response artifacts.

## Observability

Logs and metrics include correlation IDs without leaking secrets or restricted
data. Operational telemetry is not scientific evidence unless collected by a
declared evidence method.

