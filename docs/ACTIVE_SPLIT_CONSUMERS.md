# Active Split Consumers

## Summary

This document inventories the active repository code paths that consume train, validation, calibration, or test data. The authoritative split source is the regenerated manifest in [metadata/dataset_manifest.csv](../metadata/dataset_manifest.csv).

## Inventory

| File | Function/class | Line range | Current source of split assignment | Active/Legacy | Risk | Required correction |
| --- | --- | --- | --- | --- | --- | --- |
| [data/dataset.py](../data/dataset.py) | MIMIIDataset._scan | around 80-120 | Shared manifest loader via [utils/split_utils.py](../utils/split_utils.py) | Active | Low | Keep using the shared manifest interface; do not re-derive splits from the filesystem. |
| [data/dataset.py](../data/dataset.py) | get_dataloaders | around 220-250 | Shared manifest loader via dataset splits | Active | Low | Continue to use the manifest-defined train/validation rows only. |
| [data/dataset.py](../data/dataset.py) | get_normal_loader | around 250-270 | Shared manifest loader via train split | Active | Low | Continue to use normal train rows only for calibration fitting. |
| [train.py](../train.py) | main | around 40-50 | DataLoader factory from [data/dataset.py](../data/dataset.py) | Active | Low | Keep using train/validation loaders only; no test rows should be loaded. |
| [training/trainer.py](../training/trainer.py) | Trainer.fit / _validate | around 100-230 | Validation loader passed from [train.py](../train.py) | Active | Low | Validation should remain validation-only; checkpoint selection should not use test rows. |
| [calibrate.py](../calibrate.py) | main | around 60-100 | get_normal_loader from [data/dataset.py](../data/dataset.py) | Active | Low | Keep calibration fit on normal train rows only. |
| [evaluate.py](../evaluate.py) | main | around 30-60 | MIMIIDataset(cfg, split="val") | Active | Low | This remains validation evaluation; final test evaluation should be added as a separate explicit mode. |
| [inference/detector.py](../inference/detector.py) | AnomalyDetector.fit_reference_distribution | around 80-140 | Loader supplied by calibration path | Active | Low | Ensure it receives only train-derived normal samples. |
| [utils/split_utils.py](../utils/split_utils.py) | load_manifest_split | around 70-140 | Regenerated manifest | Active | Low | This is the single authoritative split interface. |
| [scripts/generate_dataset_manifest.py](../scripts/generate_dataset_manifest.py) | generate_manifest / assign_split | around 80-120 and 150-200 | Deterministic machine-ID bucket assignment | Active | Medium | Keep manifest generation authoritative; avoid duplicate split logic elsewhere. |
| [tests/test_data_integrity.py](../tests/test_data_integrity.py) | test_active_train_loader_uses_manifest_train_rows, etc. | around 150-220 | Shared manifest loader and dataset loaders | Active | Low | Continue to enforce the pipeline end to end. |
