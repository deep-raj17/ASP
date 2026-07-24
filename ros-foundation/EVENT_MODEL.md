# ROS Event Model

Specification: ROS-FS-01  
Version: Draft 1.0

## Event envelope

Every event contains:

- event ID and versioned type;
- aggregate ID/type and monotonically increasing aggregate sequence;
- registry ID/position;
- occurred-at and recorded-at UTC timestamps;
- producer identity/version;
- correlation, causation, and idempotency IDs;
- project namespace and policy context;
- payload schema/version and payload or artifact digest;
- previous event/entry digest and signature metadata.

Events are immutable facts in past tense. Commands are requests and are never
published as completed events.

## Canonical events

| Event | Producer | Typical consumers | Required payload |
|---|---|---|---|
| `ExperimentStarted.v1` | Experiment Runner | registry, telemetry | experiment/attempt, inputs, environment |
| `ExperimentCompleted.v1` | Experiment Runner | evidence, statistics | outputs, exit status, completeness |
| `EvidenceRegistered.v1` | Evidence Engine | verifiers, lineage | evidence ID, subject, artifacts |
| `EvidenceVerified.v1` | Verifier coordinator | gate engine, claims | verification ID, method, result |
| `GateSatisfied.v1` | Gate evaluator | workflow, dashboard | gate evaluation and exact inputs |
| `GateUnsatisfied.v1` | Gate evaluator | workflow, audit | failed predicates and inputs |
| `WorkflowBlocked.v1` | Workflow engine | interfaces, notifications | blocker codes and requirements |
| `WorkflowFailed.v1` | Workflow engine | interfaces, audit | terminal rule/evidence |
| `WorkflowResumed.v1` | Workflow engine | task dispatch | prior blocker and new position |
| `ArtifactReleased.v1` | Release Builder | publication, archive | release ID, manifest digest, approval |
| `PublicationSubmitted.v1` | External integration | publication registry | external receipt artifact |
| `ReviewReceived.v1` | External integration | RMS workflow | source, decision/comment artifact |

Simulated reviews use `SimulatedReviewCompleted.v1`, never `ReviewReceived.v1`.

## Ordering

Ordering is guaranteed only per aggregate sequence. Cross-aggregate consumers
use causation/correlation and tolerate reordering. A consumer waits or parks an
event when a required predecessor is missing.

## Delivery and idempotency

Transport is at-least-once. Event IDs are globally unique; consumers persist
processed IDs and last aggregate sequence atomically with their effects.
Duplicate payload with a new ID is not automatically equivalent and is flagged.

## Failure handling

Transient consumer failures retry under policy. Permanent schema or integrity
failures enter a dead-letter registry and emit an incident. Replay requires
authorization, preserves original IDs, and records replay attempts. Poison
events never get silently skipped.

## Evolution

Event type major versions are distinct types. Producers may dual-publish during
a bounded migration. Consumers declare supported versions. Upcasters are
versioned deterministic projections and never rewrite stored events.

