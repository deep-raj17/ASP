# CHAAD Re-import Guide

1. Configure private paths outside Git.
2. Validate the PMPS definition and project manifest.
3. Run `python -m ros.cli.main --config . --dry-run project add
   projects/chaad/project.yaml`.
4. Run `python -m projects.chaad.import.import_chaad --source . --workspace .
   --dry-run`.
5. Review checksums, derived state, and registry integrity.
6. Run the same commands without `--dry-run` only with approval.
7. Re-run once; every existing evidence and registry record must be idempotent,
   log counts must remain unchanged, and PMPS-01 must derive the same state.
8. Run `python -m ros.cli.main --config . --project chaad registry verify`
   and `python -m ros.cli.main --config . --project chaad status`.

Never delete `.ros` history to make a re-import pass. Changed source content
under an existing evidence ID must fail as a duplicate conflict and be
registered under a new versioned identity.
