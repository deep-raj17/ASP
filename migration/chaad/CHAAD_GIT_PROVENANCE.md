# CHAAD Git and Source Provenance

- Captured HEAD: `9a9cd3929328d8480712face6265f04320dc971d`
- Branch: `blackboxai/research-integrity-audit`
- Remote identity: `https://github.com/deep-raj17/ASP` (credentials absent)
- ROS tags present: `ros-core-01-v1.0.0`, `ros-core-02-v1.0.0`,
  `ros-core-03-v1.0.0`, `ros-cli-01-v1.0.0`
- Working tree at adapter inventory: DIRTY — 69 staged entries, 13 entries
  with unstaged changes, and 47 untracked entries.

The dirty state predates the adapter and is preserved. No clean/reset/checkout
operation was performed. The captured PMPS-01 Git snapshot remains evidence of
an earlier state and is not rewritten. No claimed CHAAD release or publication
tag was found. The four ROS milestone commits and annotated tags were verified
locally; no push was performed.

Relevant history:

- `9a9cd39` safe ROS CLI
- `aa3f95b` immutable ROS registries
- `ea346d3` evidence/provenance engine
- `142c78e` workflow/state engine
- `686c450` CHAAD workspace CLI and review-script correction

Uncommitted user work remains outside the scoped ROS commits.
