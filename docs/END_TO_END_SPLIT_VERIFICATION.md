# End-to-End Split Verification

## Verification summary

- The manifest is the authoritative source of train/validation/test assignments.
- The active dataset loaders use the shared manifest-driven split interface.
- The regression suite passes and verifies that active loaders consume the expected manifest rows.

## Evidence

- Manifest path: [metadata/dataset_manifest.csv](../metadata/dataset_manifest.csv)
- Shared interface: [utils/split_utils.py](../utils/split_utils.py)
- Active loaders: [data/dataset.py](../data/dataset.py)
- Regression command: `c:/ASP/ASP/.venv/Scripts/python.exe -m pytest tests/test_data_integrity.py -q`
- Exit code: 0
- Passed tests: 29
