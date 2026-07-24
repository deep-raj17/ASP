# ROS-FS-01 Architectural Review

Specification: ROS-FS-01  
Version: Draft 1.0.1

## Review result

**PASS FOR IMPLEMENTATION DESIGN**, subject to the outstanding risks below.
The foundation is internally consistent, specification-first,
project-independent, evidence-driven, append-only, and implementable without a
foundational architectural redesign.

## Completeness review

- Layers and dependency direction are explicit.
- Domain identities, aggregate boundaries, and lineage are defined.
- Workflow, gate, evidence/verification, approval, experiment, and publication
  states are distinct.
- All requested schema families have fields, validation rules, examples, and
  compatibility policy.
- Workflow execution, blocking/failure, retry/resume, and compensation are specified.
- Evidence cannot contain or approve a gate verdict.
- Registries and events are append-only, hash-chained, and replayable.
- Modules are discoverable, versioned, permission-bounded capabilities.
- Agents cannot approve gates or rewrite evidence/history.
- CLI, events, security, quality attributes, and implementation phases align.

## Circular dependency review

No normative circular layer dependency exists. Workflow and Evidence cooperate
through core contracts and events rather than importing each other. Registries
store their facts but do not call producers. Interfaces depend inward only.

## Ambiguities resolved

- Missing evidence is `BLOCKED`; verified failed criteria are `UNSATISFIED` and
  may cause workflow `FAILED`.
- Experiment execution success is separate from hypothesis support.
- Human approval authorizes action but does not satisfy scientific evidence.
- `NOT_APPLICABLE` is a verified applicability result, never a skipped gate.
- `WAIVED` is a policy-authorized administrative exception with an approval
  reference; it is distinct from scientific satisfaction.
- Simulated reviews cannot create external review/publication states.
- Corrected evidence supersedes rather than mutates.

## Outstanding risks

1. The machine-readable constraint language is not selected; it must remain
   deterministic, sandboxed, and auditable.
2. Identity, signature, and trust-root providers require deployment-specific design.
3. Privacy-driven erasure must reconcile legal requirements with append-only integrity.
4. Distributed registry replication and disaster recovery need formal consistency targets.
5. Confidence semantics require domain-specific verifier profiles.
6. Event payload size, artifact storage, and long-term cost need capacity modeling.
7. Project adapter sandboxing and Windows/Linux path portability need prototypes.
8. Compliance packs require legal/venue review and cannot be inferred by software.

## Future extension points

Federated registries, institutional trust federation, confidential computing,
reproducible compute attestations, richer provenance standards, domain-specific
workflow packs, and read-only public evidence portals fit existing contracts.

## Recommendation

Freeze ROS-FS-01 Draft 1.0 for stakeholder review. Resolve the outstanding
policy-language, trust, privacy, and registry-consistency ADRs before Phase 2.
Implementation must begin with schemas and conformance fixtures, not UI or agents.
