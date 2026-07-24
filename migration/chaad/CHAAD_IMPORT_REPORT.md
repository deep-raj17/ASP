# CHAAD Import Report

Correlation ID: `ros-project-01-chaad-20260724`

The project CLI dry-run validated `project-chaad-v1`, followed by real
registration through `ros project add`. The adapter dry-run executed the full
sequence in a disposable ROS workspace.

The initial real import registered 17 immutable evidence records with 28 verifier
executions and ten registry records:

- one project;
- one policy;
- one incomplete dataset and one verified split artifact;
- one incomplete/provisional experiment;
- one provisional model;
- invalid and corrected validation export artifacts;
- one incomplete publication baseline;
- one blocked workflow instance.

Registry integrity: VERIFIED, 10 records, no issues, head checksum
`584af067d299f32ad1faab246d9bd43a2169ec72135fdc52b23ef1f2225d6ff0`.

Immediate replay returned every evidence verification and registry append as
idempotent. Evidence log count remained 45 and workflow audit count remained
two. No publication-ready state or held-out test result was imported.

The remediation run then added one new full-corpus audit record after verifying
53,046 files with zero errors. PMPS-01 was re-evaluated from the expanded
evidence set and remained BLOCKED solely on authoritative local dataset
identity.
