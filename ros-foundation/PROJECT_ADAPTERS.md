# Project Adapter Specification

Specification: ROS-FS-01  
Version: Draft 1.0

## Purpose

A project adapter maps a project-owned repository and external resources into
ROS contracts. It declares, but does not relocate, project code, datasets, and
artifacts.

## Manifest content

`<project>.yaml` includes:

- project identity, owners, domain, and status;
- repository URI, commit policy, and dirty-tree handling;
- path mappings with explicit read/write/external access;
- dataset/model/config/artifact descriptors or registry references;
- pinned lifecycle/workflow/schema/policy versions;
- enabled and disabled modules with rationale;
- project-specific evidence collectors and verifiers;
- data classification, secret handling, and external access restrictions;
- supported operating systems/compute environments;
- adapter version and compatibility constraints.

## Path semantics

Paths are URIs. Portable manifests use variables resolved by a local binding
file excluded from publication. A path cannot imply authorization. Dataset
roots are read-only by default; output roots must be explicitly scoped.

## Adapter observations

Adapters may observe file existence, digests, Git state, configuration, and
project command results. Every observation becomes candidate evidence with
scope and collection time. Adapters cannot mark workflows complete.

## Example: CHAAD import

```yaml
api_version: ros.dev/v1
kind: Project
metadata:
  id: prj_chaad
  namespace: research
  created_at: 2026-07-24T00:00:00Z
  schema_version: 1.0.0
  lineage: []
spec:
  name: CHAAD
  adapter: {name: filesystem-project, version: 1.0.0}
  repositories:
    - {uri: "file:///C:/ASP/ASP", access: read, dirty_tree_policy: record}
  resources:
    dataset: {uri: "file:///E:/MIMII", access: read}
    evidence: {uri: "file:///C:/ASP/ASP/artifacts", access: read}
  lifecycle: {name: PMPS, version: 1.0.0}
  modules: {enabled: [], disabled: []}
  imported_state:
    classification: BLOCKED
    reason_code: REQUIRED_EVIDENCE_MISSING
    source: artifacts/publication_baseline/pmps01_gate.json
```

The prior `FAIL` report is imported as immutable source evidence. ROS derives
`BLOCKED` because the recorded reason is missing exhaustive verification, not
evidence that the corpus is corrupt. Import never rewrites the source report.

## Validation

Adapters fail validation for unresolved required paths, ambiguous repositories,
floating workflow versions, write access broader than policy, embedded secrets,
or incompatible schema/module declarations. Optional unavailable resources are
recorded and may block dependent workflows.

## Portability

Project identity is independent of local paths. Moving a project changes local
bindings, not the project ID. Adapter migration creates a new version with
lineage and a validation report.

