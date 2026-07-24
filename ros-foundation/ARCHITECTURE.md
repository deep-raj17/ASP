# ROS Architecture

Specification: ROS-FS-01  
Version: Draft 1.0

## Architectural invariants

1. Specifications are authoritative over implementations.
2. Registry entries and domain events are append-only.
3. Artifact bytes are content-addressed; metadata refers to their digest.
4. Evidence and verification are distinct records.
5. Gate state is derived and reproducible, never asserted by an agent.
6. Workflow state changes only through validated events.
7. Human approvals authorize actions but do not substitute for scientific evidence.
8. Interfaces call core contracts; they do not mutate registries directly.
9. Projects integrate through adapters and remain independently versioned.

## Layer model

| Layer | Purpose | Inputs | Outputs | Depends on | Versioned by |
|---|---|---|---|---|---|
| Specification Layer | Schemas, contracts, policies, workflow definitions | Governance decisions | Normative definitions | None | ROS-FS and schema versions |
| Core Layer | Identity, state machine, lifecycle, approvals, policy enforcement | Validated commands/events | Accepted events, derived state | Specification Layer | ROS core version |
| Workflow Layer | Versioned graphs, prerequisites, retries, stop rules | Workflow definitions, project bindings | Task intents, gate evaluations | Core Layer, policy | Workflow version |
| Evidence Layer | Collection references, verification, lineage, confidence | Artifacts and verifier outputs | Evidence/verification records | Core Layer, artifact engine | Evidence schema/version |
| Registry Layer | Append-only journals and projections | Accepted domain events | Historical records, read models | Core Layer, Evidence Layer | Registry format version |
| Module Layer | Optional capability discovery and execution contracts | Capability requests, events | Task results, artifacts, events | Specification Layer, Interfaces Layer | Module SemVer |
| Agent Layer | Delegated task execution under bounded authority | Task intent, permissions | Candidate artifacts/results | Policy, Module Layer | Agent contract/version |
| Automation Layer | Experiment, statistics, paper, artifact, release services | Authorized task intents | Artifacts and evidence candidates | Module Layer, event bus | Service version |
| Interfaces Layer | CLI, REST, dashboard, IDE and source-control integrations | User/system commands | Stable responses and exit codes | Core Layer contracts | Interface version |
| Projects Layer | Project manifests, paths, enabled modules, workflow binding | Local/external project state | Adapter observations and references | Specification Layer, policy | Adapter version |

## Dependency direction

```text
Interfaces ─┐
Agents ─────┼─> Core contracts ─> Workflow/Evidence/Policy ─> Registries
Automation ─┘          ↑                     ↑
Modules ───────────────┘                     │
Projects ── validated adapters ──────────────┘
                 all constrained by Specifications
```

No lower layer imports an interface. Registries do not invoke modules.
Modules communicate through commands and events, not direct registry writes.

## Control and data planes

- **Control plane:** commands, workflow state, policies, approvals, events.
- **Evidence plane:** immutable artifacts, checksums, verification records.
- **Execution plane:** external compute and module/agent task execution.
- **Projection plane:** rebuildable status views, dashboards, reports.

Only append-only records are authoritative. Projections and dashboards are
disposable and rebuildable.

## Primary contracts

- Command envelope: intent, actor, project, expected version, idempotency key.
- Event envelope: identity, aggregate, sequence, causation, correlation, payload.
- Artifact descriptor: digest, media type, size, storage locator, access policy.
- Evidence record: claim-independent observation linked to artifacts.
- Verification record: verifier, method, result, scope, policy context.
- Gate definition: predicates over verification records and project facts.
- Project adapter: declared paths/capabilities with read/write boundaries.

## Deployment boundary

ROS may be deployed as a local CLI, a single service, or distributed services.
The observable contracts and ordering semantics remain identical. Storage and
transport choices are implementation decisions provided they preserve
append-only history, atomic expected-version writes, and deterministic replay.
