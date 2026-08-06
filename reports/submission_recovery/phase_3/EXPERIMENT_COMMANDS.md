# Experiment Commands

All commands run from `C:\ASP\ASP`.

## Safe inspection

```powershell
python train.py --help
python train.py --submission-run --run-id phase3-contract-dryrun-20260727 --phase 4 --seed 42 --epochs 4 --dry-run
```

## Phase 4 pilot commands

These are authorized only after the Phase 3 PASS decision:

```powershell
python train.py --submission-run --run-id phase4-chaad-seed42 --phase 4 --seed 42 --epochs 4
```

The simple neural pilot requires its dedicated frozen configuration runner; it
must not be approximated by silently changing the CHAAD architecture.

## Validation evaluation

```powershell
python evaluate.py --split validation --phase 4
```

## Validation-only baseline extraction

```powershell
python scripts/run_baselines.py --phase 5 --checkpoint <run-checkpoint> --output <create-only-output>
```

## Reliability-aware grouped CV

```powershell
python scripts/run_reliability_cv.py --features <validation-features.csv> --predictions-output <create-only-predictions.csv> --report-output <create-only-report.json> --seed 42 --phase 5
```

## Forbidden example

```powershell
python evaluate.py --split test --phase 3
```

This exits with `PermissionError` before dataset construction.
