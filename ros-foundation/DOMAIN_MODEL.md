# ROS Domain Model

Specification: ROS-FS-01  
Version: Draft 1.0

## Identity

Every entity has a globally unique opaque `id`, a `kind`, a schema version,
creation event, owning namespace, and lineage. IDs are never reused. Human
labels are mutable projections and are not identity.

## Entities

| Entity | Definition | Required attributes | Key relationships | Constraints and lifecycle |
|---|---|---|---|---|
| Project | Research boundary managed by ROS | id, name, adapter, owner, workflow binding | datasets, models, experiments, publications | Adapter changes append versions |
| Workflow | Versioned directed execution graph | id, version, nodes, edges, gates | project, policies, modules | Definition immutable after release |
| Gate | Derived decision boundary | id, predicates, required evidence, policy | workflow node, verification records | No agent-authored verdict |
| Evidence | Immutable observation descriptor | id, subject, type, artifact refs, provenance | artifacts, experiment, verifier | Never contains gate state |
| Verification | Assessment of evidence | id, evidence id, verifier, method, result | evidence, policy, gate evaluation | Append-only; may be superseded |
| Artifact | Content-addressed byte sequence or external immutable object | digest, algorithm, size, media type, locator | evidence, experiment, publication | Digest verified before use |
| Dataset | Versioned dataset identity and declared partitions | id, version, manifest, license, digest | project, experiments | Split identity cannot mutate in place |
| Model | Versioned model definition or checkpoint | id, version, architecture ref, artifact | experiments, evaluations | Checkpoint bytes content-addressed |
| Experiment | Controlled scientific execution | id, hypothesis, config, inputs, state | project, dataset, model, artifacts | Negative/failed runs retained |
| Publication | Manuscript lifecycle aggregate | id, version, venue, state, claims | evidence, artifacts, reviews | Numerical changes require lineage |
| Review | External or simulated review record | id, source, received time, artifact | publication, revisions | External vs simulated is mandatory |
| Policy | Versioned rule set | id, version, scope, rules, severity | actions, gates, approvals | Released policy immutable |
| Approval | Human authorization record | id, request, principal, decision, scope | policy, command, project | Cannot satisfy evidence predicates |
| Module | Discoverable optional capability | id, version, capabilities, compatibility | workflows, gates, events | Installed version immutable |
| Agent | Non-authoritative task executor | id, type, permissions, provider | task, module, artifacts | Cannot approve gates or rewrite history |
| Registry | Append-only journal for a domain | id, kind, format version | events, projections | Expected-version concurrency |
| Lifecycle | Named collection of versioned workflows | id, version, entry/exit policy | projects, workflows | Released topology immutable |
| Event | Immutable fact that occurred | id, type, aggregate, sequence, payload | registry, causation chain | Idempotent, ordered per aggregate |
| State | Replay-derived aggregate condition | name, source event position | workflow/gate/etc. | Not independently writable |
| Command | Requested action | id, actor, intent, expected version | policy, approval, resulting events | May be rejected without event mutation |
| Claim | Scientific statement requiring support | id, text, scope, evidence requirements | publication, gates, evidence | Support state is derived |

## Relationship rules

- A Project selects one released Lifecycle version and pins Workflow versions.
- An Experiment references exact Dataset, configuration, code, environment, and
  Model versions; missing references yield `BLOCKED`, not inferred values.
- Evidence refers to one or more Artifacts and a subject; Verification evaluates
  Evidence within a declared scope.
- Gate evaluation consumes Verification records and project facts at a registry
  position. Its input set and evaluator version are recorded.
- Publication claims trace through Claim → Gate evaluation → Verification →
  Evidence → Artifact/Experiment.
- Approval authorizes a command but does not alter verification results.

## Aggregate boundaries

Project, WorkflowRun, Experiment, Publication, ApprovalRequest, and Registry are
transactional aggregates. Cross-aggregate effects use events and idempotent
consumers; no distributed transaction is assumed.

## Constraint language

Normative constraints use `MUST`, `MUST NOT`, `SHOULD`, and `MAY` as defined by
RFC 2119. Machine-evaluable rules are declared in versioned policy or gate
definitions; prose cannot override them.

