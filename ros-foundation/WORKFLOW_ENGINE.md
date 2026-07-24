# Workflow Engine Specification

Specification: ROS-FS-01  
Version: Draft 1.0

## Execution model

A workflow definition is an immutable, versioned directed graph. A WorkflowRun
pins the definition, project adapter, policy bundle, module resolutions, and
starting registry positions. Nodes emit task intents; gates determine whether
edges become traversable.

The engine is a deterministic interpreter. Given identical definitions,
registry histories, policy versions, verified artifacts, and evaluation time,
it must derive the same state.

## Command lifecycle

1. Validate schema and actor identity.
2. Enforce expected aggregate version and idempotency.
3. Resolve exact workflow/module/policy versions.
4. Evaluate permissions and approval requirements.
5. Check prerequisites and stop conditions.
6. Append accepted command event.
7. Dispatch task intent or evaluate gate.
8. Append outcomes and update rebuildable projections.

## Gate evaluation

A gate evaluator records definition digest, input evidence and verification
IDs, policy versions, registry positions, evaluator version, time, and verdict.
Missing required input yields `BLOCKED`; a verified counterexample yields
`UNSATISFIED`. Gate evaluation cannot execute arbitrary module code.

## Dependencies and parallelism

Nodes may run concurrently only when:

- all declared incoming dependencies are satisfied;
- resource and policy constraints allow it;
- neither node mutates the same external target without declared serialization;
- outputs have unique artifact identities.

Join nodes define `all`, `any`, or quorum semantics. Quorum is forbidden for
mandatory scientific evidence unless the gate definition explicitly justifies it.

## Stop conditions

Stop rules are evaluated before dispatch and after every accepted event.
Severity:

- `pause`: workflow becomes `BLOCKED`;
- `fail`: workflow becomes `FAILED`;
- `cancel`: requires authorized cancellation;
- `emergency_stop`: halts dispatch and requests incident review.

No stop rule deletes already produced artifacts.

## Retry and resume

Retry policy declares retryable error codes, maximum attempts, backoff,
idempotency behavior, and compensation requirements. Scientific result
failures are not infrastructure retries.

Resume:

- revalidates the project adapter and policy bundle;
- verifies prior artifacts by digest;
- reevaluates blockers at a named registry position;
- dispatches only incomplete idempotent intents;
- creates new attempts without rewriting old attempts.

## Rollback and compensation

Append-only scientific history cannot roll back. Reversible external operations
may define compensating commands. Compensation is recorded and cannot erase the
original event. Published artifacts require release-policy handling, not rollback.

## Workflow versioning

A run never silently upgrades. Forking to a newer workflow creates a new run
with migration lineage and a compatibility report. Historical projects may
continue on supported old versions.

## Failure taxonomy

- Validation failure: rejected command; no domain mutation.
- Dependency block: `BLOCKED`.
- Scientific criterion not met: gate `UNSATISFIED`; workflow policy decides failure.
- Execution failure: attempt `FAILED`; retry may be allowed.
- Integrity incident: quarantine evidence, block dependent runs, emit incident.
- Policy violation: deny action and record an audit decision.

