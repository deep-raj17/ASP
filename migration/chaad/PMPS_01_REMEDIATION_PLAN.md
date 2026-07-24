# PMPS-01 Remediation Plan

## R-001 — Full-corpus integrity

- Priority: critical
- Requirement: `dataset_integrity`
- Problem: the earlier report decoded only one of 53,046 WAV files.
- Evidence needed: a new non-overwriting report proving live readability,
  finite samples, metadata agreement, and current SHA-256 for every row.
- Safe command: `python scripts/audit_pmps01_dataset.py --manifest
  metadata/dataset_manifest.csv --output
  migration/chaad/remediation/pmps01_dataset_audit_20260724.json`
- Dry-run: two-file real-data smoke test completed before the full read.
- Approval: user explicitly requested following the remediation plan.
- Scope: 135,802,003,680 bytes, 53,046 files; read-only.
- Dependency: local dataset availability.
- Completion: all five counters equal 53,046 and error count is zero.
- Prohibited shortcuts: sampling, trusting manifest hashes without current
  recomputation, modifying WAVs, or overwriting the historical report.
- Execution: **COMPLETED and VERIFIED** on 2026-07-24. All five counters equal
  53,046; 135,802,003,680 bytes were read; error count is zero. The final
  report supersedes neither nor overwrites the historical partial report.

## R-002 — Authoritative local dataset identity

- Priority: critical
- Requirement: `dataset_license_identity`
- Problem: the official MIMII record states public 1.0 and CC BY-SA 4.0, and
  its four IDs/machine types/noise folders agree with the local structure, but
  the extracted local files cannot be tied cryptographically to the twelve
  official ZIP checksums.
- Evidence needed: original ZIP files matching the Zenodo MD5 values, a
  trustworthy download receipt, or authoritative per-file hashes.
- Safe command: hash locally retained original archives if supplied; do not
  re-download or reconstruct archives without approval.
- Dry-run: available for archive-registration planning.
- Approval: download or storage of ~100 GB requires explicit approval.
- Scope: external provenance, no experiment or dataset mutation.
- Dependency: original acquisition evidence or archives.
- Completion: authoritative release identity and license linkage verified.
- Prohibited shortcuts: treating matching folder names, dates, or Markdown as
  cryptographic identity proof.
- Execution: official release/license located; local identity remains BLOCKED.

The gate must be machine-re-evaluated after new evidence registration. It must
not be manually set to satisfied.
