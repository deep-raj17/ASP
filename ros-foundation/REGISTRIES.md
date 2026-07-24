# Append-Only Registry Specification

Specification: ROS-FS-01  
Version: Draft 1.0

## Registry families

ROS defines registries for Projects, Datasets, Models, Experiments, Artifacts,
Publications/Papers, Reviews, Policies, Approvals, Modules, and Workflows.

## Entry envelope

Every entry contains:

- registry ID and monotonically increasing sequence;
- entry ID, entity ID, and event type;
- schema and registry format versions;
- UTC timestamp and attributable author/principal;
- correlation, causation, and idempotency IDs;
- previous-entry digest and current-entry digest;
- payload or immutable payload artifact reference;
- parent lineage and project namespace;
- signature metadata where policy requires it.

## Append semantics

Writes use compare-and-append with expected sequence. Conflicts are rejected and
retried after reread. Entry identity and idempotency key prevent duplicates.
Corrections append a correcting event; deletion and in-place update are absent
from the contract.

## Identity and version

Entity IDs are stable. Entity versions are reconstructed from their event
stream. Semantic versions describe released definitions/artifacts, while event
sequence describes history; they are not interchangeable.

## Integrity

Each registry is hash-chained. Periodic signed checkpoints may anchor the chain
externally. Verification checks sequence continuity, previous digests, payload
digests, signatures, and schema support. Broken chains quarantine later entries
until investigated.

## Projections and snapshots

Status views, indexes, dashboards, and “current” records are derived
projections. Snapshots accelerate replay but contain source registry position,
projection version, and digest. They can be discarded and rebuilt.

## Lineage

Parent lineage is explicit across registries. An experiment points to exact
project, dataset, model/config/code, workflow, and policy versions. A
publication points to exact claims, experiments, evidence, tables, and figures.

## Retention and archival

Scientific registry entries are retained indefinitely unless a higher legal
policy requires restricted tombstoning. Tombstones hide access while preserving
identity, reason, approval, and integrity chain. Archive moves storage, not
history, and remains verifiable.

## Import/export

Portable exports include entries, schemas, digests, signatures, and a manifest
of registry positions. Import creates a new import lineage and verifies the
bundle; it never merges histories by overwriting IDs.

