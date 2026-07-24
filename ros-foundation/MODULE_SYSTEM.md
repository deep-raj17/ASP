# Module System Specification

Specification: ROS-FS-01  
Version: Draft 1.0

## Purpose

Modules provide optional, reusable capabilities without changing the core.
Examples include benchmarking, statistics, explainability, deployment,
commercialization, grants, and thesis support.

## `module.yaml` contract

Required fields:

```yaml
api_version: ros.dev/v1
kind: Module
metadata:
  id: module_statistics
  namespace: official
  created_at: 2026-07-24T00:00:00Z
  schema_version: 1.0.0
  lineage: []
spec:
  name: statistics
  version: 1.2.0
  publisher: ros-foundation
  capabilities: [statistical_tests, confidence_intervals]
  requires:
    capabilities: [experiment_registry]
    modules: []
  inputs: [experiment-result/v1]
  outputs: [statistical-report/v1]
  consumes_events: [ExperimentCompleted.v1]
  emits_events: [StatisticalAnalysisCompleted.v1]
  supports:
    workflows: [{name: PMPS, range: ">=1.0.0 <2.0.0"}]
    gates: [metrics-statistically-verified]
  permissions: [artifact.read, artifact.create, evidence.propose]
  configuration_schema: statistical-module-config/v1
  lifecycle: stable
```

Before execution, ranges resolve to exact versions and the resolution is
recorded.

## Capability discovery

The module catalog is an append-only registry. Discovery filters by capability,
schema compatibility, workflow/gate support, project policy, platform, and
permissions. Selection is deterministic using pinned resolution policy; ties
require explicit project choice.

## Inputs and outputs

Modules consume validated descriptors, never mutable filesystem assumptions.
Outputs are candidate artifacts and events. Only the evidence engine may accept
them as evidence; only gate evaluation may derive a verdict.

## Dependencies

Module dependencies form an acyclic graph. Transitive capabilities and
permissions are not implicit. Conflicts fail resolution before workflow
execution. Optional dependencies must have explicit degraded behavior.

## Lifecycle

States: `experimental`, `preview`, `stable`, `deprecated`, `retired`,
`quarantined`. Deprecation provides replacement and support dates. Retirement
does not invalidate historical runs. Quarantine blocks new execution but
preserves artifacts and history.

## Isolation and security

Modules declare filesystem, network, compute, secret, and external-system
permissions. The runtime grants the minimum intersection of module request,
project policy, and actor authority. Modules cannot write registries directly,
approve gates, or expand their permissions.

## Compatibility

Compatibility is checked across module, core, schemas, workflows, events, and
project adapter. Major event/schema changes require a new consumer declaration.
Compatibility evidence is recorded for every resolved run.

