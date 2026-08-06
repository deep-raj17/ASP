# PMPS-01 Recovery Report

**Decision: BLOCKED — conflicting evidence**

The PMPS-01 workflow specification was reconstructed directly from
`ros/specs/workflows/pmps-1.0.0.yaml`. Twelve requirements have verified local
evidence. `dataset_license_identity` remains incomplete: the extracted local
corpus has conflicting archive evidence: five local archives match official
MD5s, one (`0_dB_valve.zip`) mismatches, and three official archives are absent.
Formal adjudication classifies this as Category C: repository evidence is
conflicting.

The canonical extracted root is nevertheless internally consistent: all 53,046
manifest files are present, readable, finite, metadata-consistent, and match
their recorded current-file SHA-256 values. This does not resolve archive
identity.

The full-corpus audit itself is verified (53,046 files, zero reported errors),
but local readability and current-file hashes do not establish acquisition
identity. The workflow therefore correctly refuses to advance. PMPS-02 through
PMPS-08 remain unevaluated.

Required recovery:

1. Resolve the `0_dB_valve.zip` mismatch and recover the three missing official
   archives or equivalent authoritative acquisition evidence.
2. Register that evidence without altering raw data or the manifest.
3. Re-evaluate PMPS-01 through the ROS workflow engine.

The provenance certification and reconciliation artifacts are under
`research_validation/provenance/` and `research_validation/leakage/`. A frozen
experiment protocol draft exists under `research_validation/protocol/`, but
all experiment rows remain `PROPOSED` and unauthorized.

No training, test-set evaluation, package installation, or publication claim
was performed as part of this recovery analysis.
