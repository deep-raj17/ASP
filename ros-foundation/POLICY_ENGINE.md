# Policy, Approval, Permission, and Compliance Specification

Specification: ROS-FS-01  
Version: Draft 1.0

## Separation of concerns

- **Policy:** evaluates rules over subject, action, resource, context, and risk.
- **Permission:** establishes what an authenticated principal may request.
- **Approval:** records human authorization for a specific action digest.
- **Compliance:** checks external/internal obligations and produces evidence.

These subsystems share contracts but remain independently versioned.

## Decision model

Policy outputs `ALLOW`, `DENY`, `REQUIRE_APPROVAL`, `REQUIRE_EVIDENCE`, or
`NOT_APPLICABLE`, plus reason codes and policy digests. Explicit deny wins.
Decisions are deterministic for the same inputs and evaluation time.

## Risk classes

| Class | Examples | Default handling |
|---|---|---|
| R0 Read-only | status, verify digest | Allow with audit |
| R1 Reversible project write | new report in scoped output | Permission required |
| R2 Scientific mutation | training, threshold selection | Approval + protocol |
| R3 External/material | submission, release, messaging | Explicit human approval |
| R4 Destructive/regulated | delete evidence, expose restricted data | Deny or exceptional multi-party approval |

## Mandatory policies

- Evidence and registry history cannot be overwritten or deleted.
- Agents cannot approve gates, approvals, submissions, or releases.
- Test/holdout access follows project data-use policy.
- Results cannot be changed without new experiment/evidence lineage.
- External publication states require external evidence.
- Secrets, personal data, and licensed datasets follow access classification.
- Separation of duties applies to high-risk approval and execution.

## Approval contract

An approval binds request digest, action, target, scope, approver role,
conditions, expiry, and maximum uses. Changed parameters invalidate approval.
Approvals may authorize execution but cannot turn rejected evidence into
verified evidence or an unsatisfied gate into a satisfied gate.

## Compliance packs

IEEE/ACM artifact, venue, institutional, data-license, security, and
reproducibility requirements are versioned policy packs. A project pins exact
packs. Compliance output is evidence with coverage and exceptions, not a
marketing badge.

## Forbidden actions

Default-denied actions include rewriting registries, deleting failed
experiments, unpinned workflow upgrades, bypassing gates, fabricating citations,
changing artifact bytes under an existing digest, and delegating broader
permissions than the delegator owns.

## Appeals and exceptions

Exceptions require a versioned exception policy, documented rationale,
time-bounded approval, compensating controls, and audit event. Non-negotiable
integrity rules have no exception path.

