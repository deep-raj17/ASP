# Security and Audit Specification

Specification: ROS-FS-01  
Version: Draft 1.0

## Security objectives

Protect integrity, attribution, confidentiality, availability, and traceability
without confusing security controls with scientific validity.

## Identity and permissions

Principals use authenticated identities. Authorization combines role,
project/resource scope, action, risk, environment, and policy version. Service
and agent identities are distinct from humans. Least privilege and
separation-of-duty are mandatory for high-risk operations.

## Audit logging

Audit records include authentication outcome, command digest, policy decision,
approval, affected identities, before/after aggregate versions, result code,
correlation ID, and time. Sensitive values are redacted by typed schema, not
free-text filtering. Audit logs are append-only and independently retained.

## Integrity and tamper detection

- Content-address all artifacts.
- Hash-chain registries and audit journals.
- Verify digests on ingest, retrieval, verification, export, and release.
- Sign high-risk approvals, registry checkpoints, and release manifests.
- Periodically anchor signed roots outside the primary store.
- Quarantine mismatches and recursively identify dependent claims/gates.

## Secrets and restricted data

Manifests contain secret references, never secret values. Providers handle
rotation and access logging. Dataset and artifact classifications determine
encryption, residency, retention, export, and redaction. Portable bundles omit
restricted bytes unless explicitly approved.

## Supply chain

Modules, verifiers, agents, and automation services are identified by package or
container digest, publisher, signature, dependency manifest, and provenance.
Untrusted modules run isolated with no implicit network or filesystem access.

## Threats

Controls address registry rewriting, artifact substitution, forged verifier
output, agent privilege escalation, replayed approval, event duplication,
dependency compromise, data exfiltration, malicious project adapters, and
confusion between simulated and external publication events.

## Incident handling

An integrity/security incident emits a typed event, quarantines affected
artifacts, blocks dependent workflows, preserves forensic evidence, identifies
lineage impact, and requires authorized resolution. Resolution appends findings
and remediation; it does not erase the incident.

## Trust boundaries

External compute, source control, object stores, publication portals, identity
providers, and agent providers are distinct trust zones. Every crossing uses a
recorded descriptor, digest where possible, and explicit policy.

