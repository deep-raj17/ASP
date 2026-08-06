# Phase 3 Dry-Run Report

## Isolated training contract

Command:

```powershell
python train.py --submission-run --run-id phase3-contract-dryrun-20260727 --phase 4 --seed 42 --epochs 4 --dry-run
```

Result: PASS. The command verified dataset-root availability, frozen seed and
epoch bounds, train/validation access, unique output paths, disabled implicit
resume, and created an immutable contract without loading audio or training.

## Validation prediction evidence

`artifacts/EXP-CHAAD-001/validation_predictions_corrected.csv` was checked
against manifest validation IDs:

- rows: 28,254;
- unique IDs: 28,254;
- missing IDs: 0;
- unexpected IDs: 0;
- duplicate IDs: 0;
- non-finite scores: 0;
- splits: validation only.

ROC-AUC and PR-AUC recomputed twice with exact equality:
0.6002609444987201 and 0.25788610546048196.

## Checkpoint forward

The preserved epoch-6 checkpoint loaded through restricted deserialization.
On the RTX 4070 SUPER, BF16 produced finite embeddings, logits,
reconstruction, attention weights, and pooled features. FP16 produced
non-finite outputs and is prohibited by the new precision policy.

## Protected test

`python evaluate.py --split test --phase 3` exited with a protection error
before any dataset object was constructed. No protected test artifact was read.
