# ROS Schema Specifications

Specification: ROS-FS-01  
Schema family: Draft 1.0

This document defines YAML serialization contracts, not parser
implementations. Unknown fields are rejected by default. Extensions use a
namespaced `extensions` map. Timestamps are UTC RFC 3339; digests use
`algorithm:value`; versions use Semantic Versioning unless noted.

## Common envelope

Required for every document:

| Field | Rule |
|---|---|
| `api_version` | `ros.dev/<schema-major>` |
| `kind` | Exact schema kind |
| `metadata.id` | Opaque globally unique ID; immutable |
| `metadata.namespace` | Validated project or system namespace |
| `metadata.created_at` | UTC timestamp |
| `metadata.schema_version` | Exact schema version |
| `metadata.lineage` | Parent IDs/digests; empty list allowed |
| `spec` | Kind-specific body |

Optional: `metadata.labels`, `metadata.annotations`, `extensions`.

## Schema catalog

### `project.schema.yaml`

Required: name, owners, repository descriptors, adapter version, lifecycle and
workflow pins, enabled/disabled modules, policies, data classifications.
Paths must declare access (`read`, `write`, `external`) and may not contain
embedded credentials. Example:

```yaml
api_version: ros.dev/v1
kind: Project
metadata: {id: prj_chaad, namespace: research, created_at: 2026-07-24T00:00:00Z, schema_version: 1.0.0, lineage: []}
spec:
  name: CHAAD
  adapter: {name: filesystem-project, version: 1.0.0}
  repositories: [{uri: "file:///C:/ASP/ASP", access: read}]
  lifecycle: {name: PMPS, version: 1.0.0}
  workflows: [{name: PMPS-01, version: 1.0.0}]
  modules: {enabled: [], disabled: []}
  policies: [policy_research_integrity_v1]
```

### `workflow.schema.yaml`

Required: name, version, lifecycle, nodes, directed edges, entry/exit gates,
stop conditions, retry and resume policies. Graph must be acyclic unless a
cycle is explicitly bounded. Released definitions are immutable.

### `gate.schema.yaml`

Required: name, version, applicability predicate, evidence requirements,
verification predicates, combination rule, freshness, blocking/failure
classification. Predicates must be deterministic and reference declared fields.

### `evidence.schema.yaml`

Required: subject, evidence type, artifact descriptors, producer, collection
method, observed time, environment, provenance, confidentiality. Gate verdict
fields are forbidden.

### `artifact.schema.yaml`

Required: digest, size, media type, logical type, locator, creation method,
access classification. Mutable URLs require an immutable digest and retrieval
verification. Multiple locators may refer to the same digest.

### `dataset.schema.yaml`

Required: identity, version, source, license status, manifest artifact, split
artifacts, statistics evidence, preprocessing version, access policy. A changed
split or manifest creates a new dataset version.

### `experiment.schema.yaml`

Required: hypothesis, protocol, project, code digest, configuration artifact,
dataset version, seeds, environment, expected outputs, metrics, approval policy.
Result references are absent until appended by experiment events.

### `publication.schema.yaml`

Required: title, version, type, authorship, claims, manuscript artifact,
target venue, lifecycle state evidence. Submission/acceptance/publication
metadata require external artifact evidence.

### `policy.schema.yaml`

Required: name, version, scope, rules, priority, effect (`allow`, `deny`,
`require_approval`, `require_evidence`), condition, reason code. Explicit deny
wins unless a higher-order compliance policy states otherwise.

### `module.schema.yaml`

Required: name, version, publisher, capabilities, input/output schemas, emitted
and consumed events, workflow/gate compatibility, dependencies, permissions,
configuration schema, lifecycle status.

### `registry.schema.yaml`

Required: registry ID, domain kind, format version, hash-chain algorithm,
sequence rules, writer policy, retention, snapshot/projection rules. Entries
require sequence, previous digest, entry digest, author, timestamp, payload.

### `agent.schema.yaml`

Required: identity, type, provider/model or executable descriptor, capabilities,
permissions, prohibited actions, delegation rules, audit policy. `approve_gate`
and `rewrite_registry` permissions are schema-forbidden.

### `approval.schema.yaml`

Required: request digest, requested action, scope, requester, required approver
roles, separation-of-duty policy, expiry, decision records. A decision must be
digitally attributable and cannot modify evidence.

## Cross-schema validation

- Every referenced version is exact; floating ranges are allowed only during
  capability resolution and are pinned before execution.
- All artifact references resolve to a descriptor with a verified digest.
- Workflow gates reference gate definitions compatible with the workflow major.
- Module permissions are a subset of project and policy permissions.
- Experiment inputs use exact dataset/model/config/code identities.
- Registry payloads validate against the schema version recorded in the entry.
- Secret values and raw personal data are forbidden in portable manifests.

## Optional versus nullable

Missing means “not supplied”; `null` is allowed only where a schema explicitly
models “known absent.” Unknown scientific facts use a typed `unknown` reason,
not empty strings or fabricated defaults.

## Compatibility and migration

- Patch: validation clarification without changing accepted instances.
- Minor: additive optional fields or new enum values negotiated by capability.
- Major: breaking validation or semantic change.
- Readers must reject unsupported major versions and preserve unknown
  namespaced extensions when round-tripping.
- Migration creates a new document with lineage to the old document and a
  signed migration report; it never edits historical registry entries.

