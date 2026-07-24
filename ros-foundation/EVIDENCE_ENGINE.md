# Evidence Engine Specification

Specification: ROS-FS-01  
Version: Draft 1.0

## Evidence model

Evidence is an immutable description of an observation. Artifact bytes,
collection context, subject, and provenance are recorded separately from any
scientific interpretation. Verification is an append-only assessment performed
under a named method and policy.

## Collection

Collectors must record:

- producer identity and software version;
- task/experiment and project identities;
- source artifact digests;
- command/config/environment references;
- collection start/end time;
- completeness and sampling scope;
- errors, warnings, and excluded data;
- confidentiality and retention class.

Importing external evidence produces an import record and never grants verified
status automatically.

## Content integrity

Default digest: SHA-256; algorithm agility is mandatory. Large artifacts may
use a Merkle manifest whose root digest is recorded. Digest verification occurs
on ingest, retrieval, before verification, and before release.

Digital signatures bind the evidence record digest, signer identity, signing
time, and trust policy. A signature proves attribution/integrity, not truth.

## Verification

A verifier is deterministic where feasible and declares:

- method and implementation digest;
- accepted evidence types and schemas;
- input scope and preconditions;
- expected tolerances;
- result vocabulary;
- uncertainty/confidence method;
- failure and inconclusive behavior;
- freshness/expiry policy.

Human review is allowed as a typed verification source but must not be described
as machine verification. Mixed gates may require both.

## Confidence

Confidence is structured, not a free-form percentage. It includes method,
coverage, uncertainty interval where applicable, assumptions, and limitations.
Gate policies decide whether a confidence record is sufficient.

## Lineage

The lineage graph connects source data → preprocessing → experiment → result
artifact → evidence → verification → gate evaluation → claim/publication.
Cycles are invalid. Derivative evidence lists all parents and transformation
digests.

## Quarantine and supersession

Digest mismatch, malware, schema failure, or provenance break quarantines an
artifact and blocks dependent gate reevaluation. Corrected evidence is a new
record with `supersedes`; the original remains queryable. “Latest” is a
projection and cannot be used in reproducible evaluation without a pinned ID.

## Machine verification rules

- No network lookup without recorded response artifact and policy permission.
- No inferred missing values.
- Tolerances are versioned and declared before evaluation.
- Sampling evidence cannot satisfy exhaustive requirements.
- A successful process exit alone is insufficient evidence.
- Generated prose, agent assertions, and filenames are not scientific evidence.

