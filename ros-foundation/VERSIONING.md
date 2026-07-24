# ROS Versioning Strategy

Specification: ROS-FS-01  
Version: Draft 1.0

## Version domains

ROS core, foundation specification, schemas, workflows, lifecycle bundles,
modules, project adapters, policies, events, registries, and interfaces are
independently versioned and pinned in execution records.

## Semantic rules

- **Major:** incompatible semantics, validation, state, or contract behavior.
- **Minor:** backward-compatible capability or optional schema addition.
- **Patch:** compatible correction or clarification.

Draft specifications use `0.y.z` or an explicit draft label. Released workflow
definitions and policy bundles are immutable.

## Compatibility guarantees

- A core declares supported schema/workflow/module ranges.
- A WorkflowRun resolves and records exact versions before start.
- Patch upgrades may be automatic only when policy permits and reproducibility
  is unaffected; scientific workflows default to exact pins.
- Minor additions cannot change the meaning of existing fields or verdicts.
- Unsupported major versions fail closed with `ROS-E-COMPAT`.

## Historical projects

Projects retain their pinned versions for reproduction. End-of-support blocks
new runs but does not block inspection or replay in an archived environment.
Security exceptions are documented with migration options.

## Migration

A migration plan declares source/target versions, transforms, invariant checks,
lossiness, rollback/compensation, and approvals. It produces new manifests or
events with lineage and a migration evidence bundle. Registry history, evidence,
and released artifacts are never rewritten.

## Registry format

Registry format upgrades use a new reader/writer version and signed checkpoint.
The original journal remains authoritative. Rebuilt projections identify the
format and source positions used.

## Project adapter and module compatibility

Adapters declare supported project/workflow/schema versions. Modules declare
core, schema, event, workflow, and capability constraints. Resolution is
deterministic and creates a lock artifact analogous to a dependency lockfile.

## Deprecation

Deprecation notices specify replacement, rationale, first-warning version,
last-supported version, and date. No silent behavior fallback is allowed.

