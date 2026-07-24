# ROS Implementation Roadmap

Specification: ROS-FS-01  
Version: Draft 1.0

Implementation begins only after ROS-FS-01 review is accepted. Each phase has a
separate design and verification gate.

## Phase 1 — Specifications

Deliver machine-readable schemas derived from `SCHEMAS.md`, normative
terminology, fixtures, conformance tests, and accepted architectural decisions.
Exit: no unresolved foundational ambiguity.

## Phase 2 — Core Engine

Implement identity, command validation, state transition interpreter, policy
decision interface, idempotency, and audit envelope. Exit: transition/property
tests and deterministic replay pass.

## Phase 3 — Registries

Implement append-only journals, hash chains, expected-version writes,
projections, snapshots, import/export, and offline verification. Exit: fault and
tamper tests pass.

## Phase 4 — Workflow Engine

Implement definition validation, graph execution, gates, blocking/failure,
parallel joins, retries, resume, and compensation. Exit: reference workflows
replay identically.

## Phase 5 — Evidence Engine

Implement artifact descriptors, ingestion, verification coordination, lineage,
quarantine, confidence, and gate input resolution. Exit: evidence cannot
self-assert a verdict and lineage is complete.

## Phase 6 — CLI

Implement the stable command/query surface and structured errors/exit codes.
Exit: CLI conformance and dry-run safety tests pass.

## Phase 7 — Modules

Implement catalog, discovery, deterministic resolution, permission isolation,
and a minimal statistics reference module. Exit: module adds capability without
core code changes.

## Phase 8 — Automation

Add event bus/outbox, experiment runner, artifact builder, and statistics
services. Exit: duplicate delivery and partial failure are reconciled.

## Phase 9 — Interfaces

Add REST, dashboard backend, IDE/source-control integrations after CLI/core
stabilize. Exit: interfaces use only core contracts and expose registry positions.

## Phase 10 — Project Migration

Create migration tooling and import CHAAD as the first adapter. Preserve its
source `PMPS-01 FAIL` evidence while deriving `BLOCKED` for missing exhaustive
verification. Exit: import is reproducible and does not modify CHAAD.

## Cross-phase requirements

Threat modeling, documentation, compatibility testing, migration plans,
observability, and independent review accompany every phase. No dashboard,
multi-agent autonomy, or release automation precedes registry/evidence integrity.

## Initial implementation risks

Premature distributed deployment, conflating projections with truth, loose
schema semantics, unbounded policy language, verifier nondeterminism, and
over-privileged modules are the primary risks.

