# ROS State Machines

Specification: ROS-FS-01  
Version: Draft 1.0.1

States are replay-derived. A transition appends an event after policy,
expected-version, and entry-criteria validation. Historical states are never
rewritten.

## Workflow run

States: `NOT_STARTED`, `READY`, `RUNNING`, `BLOCKED`, `FAILED`, `COMPLETED`,
`CANCELLED`, `ARCHIVED`.

| From | Allowed destinations | Entry/exit rule |
|---|---|---|
| NOT_STARTED | READY, CANCELLED | Definition and project binding validate |
| READY | RUNNING, BLOCKED, CANCELLED | Required approvals and prerequisites checked |
| RUNNING | BLOCKED, FAILED, COMPLETED, CANCELLED | Node events and gates determine outcome |
| BLOCKED | READY, RUNNING, FAILED, CANCELLED | Missing prerequisite supplied or blocker confirmed |
| FAILED | READY, ARCHIVED | Retry policy permits a new attempt; history retained |
| COMPLETED | ARCHIVED | All terminal gates satisfied |
| CANCELLED | ARCHIVED | Authorized cancellation |
| ARCHIVED | none | Terminal |

`BLOCKED` means progress lacks required evidence, dependency, authority, or
resource. `FAILED` means executed evidence or a terminal policy rule proves
criteria were not met. Missing evidence alone is never `FAILED`.

## Gate

States: `UNEVALUATED`, `PENDING`, `SATISFIED`, `UNSATISFIED`, `BLOCKED`,
`WAIVED`, `NOT_APPLICABLE`.

- `UNEVALUATED → PENDING` when evaluation is requested.
- `PENDING → SATISFIED|UNSATISFIED|BLOCKED|NOT_APPLICABLE`.
- `UNEVALUATED|PENDING|BLOCKED|UNSATISFIED → WAIVED` only when the gate
  definition permits waiver and a policy-authorized, scoped approval reference
  is verified. Waiver is an administrative exception and never evidence that a
  scientific criterion was satisfied.
- Any terminal verdict may be reevaluated only as a new evaluation referencing
  a later registry position; the prior evaluation remains immutable.
- `NOT_APPLICABLE` requires a predeclared applicability predicate, verified
  facts, and policy permission. It is not a waiver or skipped gate.
- Agents and humans cannot directly set a gate state.

## Evidence and verification

Evidence availability states: `REGISTERED`, `AVAILABLE`, `QUARANTINED`,
`UNAVAILABLE`, `RETIRED`. Artifact bytes and the evidence record remain
immutable; availability changes are appended observations.

Verification states: `PENDING`, `RUNNING`, `VERIFIED`, `REJECTED`,
`INCONCLUSIVE`, `EXPIRED`.

- Verification expiry is policy-driven and does not delete its prior validity.
- A new verifier result supersedes by explicit lineage, never mutation.
- `REJECTED` means the evidence failed a declared verification, while
  `INCONCLUSIVE` means the method could not decide.

## Approval

States: `REQUESTED`, `APPROVED`, `REJECTED`, `EXPIRED`, `REVOKED`, `CONSUMED`.

Only `APPROVED` may become `CONSUMED`. Approval scope, actor, action digest, and
expiry must match. Revocation cannot undo an already completed action; it
prevents future use and emits an audit event.

## Experiment

States: `PLANNED`, `AWAITING_APPROVAL`, `APPROVED`, `QUEUED`, `RUNNING`,
`SUCCEEDED`, `FAILED`, `CANCELLED`, `ARCHIVED`.

An experiment is `SUCCEEDED` when execution completed according to protocol,
not when its hypothesis was supported. Scientific outcome is separate.
Configuration or input changes require a new experiment identity.

## Publication

States: `DRAFT`, `INTERNAL_REVIEW`, `READY_FOR_SUBMISSION`, `SUBMITTED`,
`UNDER_REVIEW`, `REVISION_REQUIRED`, `ACCEPTED`, `REJECTED`, `WITHDRAWN`,
`PUBLISHED`, `ARCHIVED`.

External transitions (`SUBMITTED`, `ACCEPTED`, `PUBLISHED`) require captured
external evidence. A simulated review can never cause an external transition.

## Forbidden transitions

- Any archived state to an active state.
- Evidence or registry history to a previous version.
- `NOT_STARTED → COMPLETED`, `DRAFT → PUBLISHED`, or `SUBMITTED → PUBLISHED` without
  required intermediate evidence.
- Gate verdict assignment by agent/human command.
- Failed experiment to succeeded; retry creates an attempt or new experiment.
- Rejected publication to accepted without a new decision artifact and lineage.

## Failure and recovery

Command validation failures append audit records but no aggregate event.
Partial external work is reconciled by idempotency key. Retries use the same
key for the same intent; changed intent requires a new key. Resume reevaluates
prerequisites at a declared registry position and never assumes prior blockers
were resolved.
