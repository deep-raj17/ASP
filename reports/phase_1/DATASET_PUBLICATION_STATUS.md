# Dataset Publication Status

## INTERNAL CORPUS INTEGRITY

**VERIFIED within the local extracted corpus.** The full-corpus audit processed
53,046 expected files and reported 53,046 readable files, 53,046 finite files,
53,046 metadata matches, 53,046 current SHA-256 matches, zero errors, and
135,802,003,680 bytes read. Evidence:
`migration/chaad/remediation/pmps01_dataset_audit_20260724.json`.

## OFFICIAL ARCHIVE LINEAGE

**CONFLICTING / NOT CERTIFIED.** The authoritative machine-readable archive
comparison contains:

- eight local archive MD5 values matching the official values;
- one mismatch: `0_dB_valve.zip`;
- three unavailable local containers: `6_dB_pump.zip`,
  `6_dB_slider.zip`, and `6_dB_valve.zip`.

This eight/one/three count is taken from
`research_validation/provenance/ARCHIVE_MD5_COMPARISON.csv`. Later narrative
documents that state five matches conflict with that raw CSV and are stale on
this count. A controlled fresh acquisition produced zero verified archives and
no fresh extracted reference corpus because transfers repeatedly stalled.
Therefore no file-level official-equivalence comparison exists.

## SPLIT INTEGRITY

**VERIFIED within the audited manifest scope.** The current manifest assigns
each machine ID to exactly one split: train `id_04` (12,045), validation
`id_00` + `id_02` (28,254), and protected test `id_06` (12,747). Cross-split
SHA duplicate checks report zero. The current manifest and sidecar bytes match
SHA-256
`7c689508cbed4d49d05ec2891b315b27722ff01a8a62b6b1c4f610e3afcd0136`,
although the history of the sidecar correction is unresolved.

## LICENSING EVIDENCE

The official MIMII Public 1.0 release identity, citation, source, and published
archive checksums are documented in the provenance records. This supports
citation and description of the publisher's terms. It does not prove that
every byte in the historical extracted corpus came from those official
archives. Any reuse must remain within the official license/terms; the
repository audit does not grant additional rights.

## PUBLICATION DISCLOSURE REQUIREMENT

The present corpus may be described as internally validated and structurally
consistent with MIMII Public 1.0, but it must **not** be described as fully
cryptographically equivalent to the official release. A paper using results
from this corpus would require a prominent provenance limitation disclosing
the mismatching/missing containers and failed fresh acquisition. Current
governance is stricter: PMPS-01 remains **BLOCKED — conflicting evidence** and
the research is frozen pending an authorized re-entry condition.
