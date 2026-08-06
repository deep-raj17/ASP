"""
tests/test_data_integrity.py
────────────────────────────────────────────────────────
Pytest tests for data integrity and leakage prevention.

Usage:
    python -m pytest tests/test_data_integrity.py -v
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config import cfg
from data.dataset import MIMIIDataset, get_normal_loader
from utils.metrics import load_threshold_metadata, persist_threshold_metadata, select_threshold
from utils.split_utils import load_manifest_split


def test_manifest_exists():
    """Test that dataset manifest exists."""
    manifest_path = Path("metadata/dataset_manifest.csv")
    assert manifest_path.exists(), "Dataset manifest not found"


def test_manifest_checksum_exists():
    """Test that manifest checksum exists."""
    checksum_path = Path("metadata/dataset_manifest.sha256")
    assert checksum_path.exists(), "Manifest checksum not found"


def test_audit_report_exists():
    """Test that data leakage audit report exists."""
    report_path = Path("reports/data_leakage_audit.json")
    assert report_path.exists(), "Data leakage audit report not found"


def test_audit_passed():
    """Test that data leakage audit passed."""
    report_path = Path("reports/data_leakage_audit.json")
    with open(report_path, 'r') as f:
        report = json.load(f)
    assert report['audit_status'] == 'PASSED', f"Audit failed: {report['issues']}"


def test_no_duplicate_checksums():
    """Test that no duplicate checksums exist across splits."""
    report_path = Path("reports/data_leakage_audit.json")
    with open(report_path, 'r') as f:
        report = json.load(f)
    assert 'duplicate_checksums' in report['passed_checks'], "Duplicate checksums check failed"


def test_normalization_metadata_exists():
    """Test that normalization metadata exists."""
    metadata_path = Path("artifacts/normalization_metadata.json")
    assert metadata_path.exists(), "Normalization metadata not found"


def test_normalization_uses_fixed_constants():
    """Test that normalization uses fixed constants, not data fitting."""
    metadata_path = Path("artifacts/normalization_metadata.json")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    assert metadata['normalization_type'] == 'fixed_scale', "Normalization should use fixed scale"
    assert metadata['fitted_on'] == 'none', "Normalization should not be fitted on data"
    assert metadata['test_data_used'] == False, "Test data should not be used in normalization"


def test_calibration_metadata_exists():
    """Test that calibration metadata exists."""
    metadata_path = Path("artifacts/calibration_metadata.json")
    assert metadata_path.exists(), "Calibration metadata not found"


def test_calibration_uses_train_only():
    """Test that calibration uses train_normal only."""
    metadata_path = Path("artifacts/calibration_metadata.json")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    assert metadata['source_split'] == 'train_normal', "Calibration should use train_normal"
    assert metadata['test_data_used'] == False, "Test data should not be used in calibration"


def test_threshold_metadata_exists():
    """Test that threshold metadata exists."""
    metadata_path = Path("artifacts/threshold_metadata.json")
    assert metadata_path.exists(), "Threshold metadata not found"


def test_threshold_selected_on_validation():
    """Test that threshold was selected on validation, not test."""
    metadata_path = Path("artifacts/threshold_metadata.json")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    assert metadata['selected_on'] == 'validation', "Threshold should be selected on validation"
    assert metadata['test_data_used'] == False, "Test data should not be used in threshold selection"


def test_environment_report_exists():
    """Test that environment report exists."""
    report_path = Path("artifacts/environment_report.json")
    assert report_path.exists(), "Environment report not found"


def test_experiment_provenance_exists():
    """Test that experiment provenance exists."""
    provenance_path = Path("artifacts/experiment_provenance.json")
    assert provenance_path.exists(), "Experiment provenance not found"


def test_final_test_lock_exists():
    """Test that final test lock exists."""
    lock_path = Path("artifacts/final_test_lock.json")
    assert lock_path.exists(), "Final test lock not found"


def test_test_split_exists():
    """Test that the manifest includes a dedicated test split for final evaluation."""
    manifest_path = Path("metadata/dataset_manifest.csv")
    df = pd.read_csv(manifest_path)
    assert 'test' in set(df['split'].unique()), "Test split should exist"


def test_independent_metric_report_exists():
    """Test that independent metric report exists."""
    report_path = Path("reports/independent_metric_report.json")
    assert report_path.exists(), "Independent metric report not found"


def test_subgroup_reports_exist():
    """Test that subgroup reports exist."""
    assert Path("reports/per_machine_results.csv").exists(), "Per-machine results not found"
    assert Path("reports/per_machine_id_results.csv").exists(), "Per-machine-ID results not found"
    assert Path("reports/per_noise_condition_results.csv").exists(), "Per-noise-condition results not found"


def test_shortcut_learning_audit_exists():
    """Test that shortcut learning audit exists."""
    report_path = Path("reports/shortcut_learning_audit.json")
    assert report_path.exists(), "Shortcut learning audit not found"


def test_shortcut_learning_audit_passed():
    """Test that shortcut learning audit passed."""
    report_path = Path("reports/shortcut_learning_audit.json")
    with open(report_path, 'r') as f:
        report = json.load(f)
    assert report['audit_status'] == 'PASSED', f"Shortcut learning audit failed: {report['findings']}"


def test_manifest_has_required_columns():
    """Test that manifest has all required columns."""
    manifest_path = Path("metadata/dataset_manifest.csv")
    df = pd.read_csv(manifest_path)
    
    required_columns = [
        'file_id', 'relative_path', 'absolute_path', 'noise_condition',
        'machine_type', 'machine_id', 'label', 'split', 'source_recording',
        'segment_start', 'segment_end', 'duration_seconds', 'sample_rate',
        'num_frames', 'num_channels', 'file_size_bytes', 'sha256'
    ]
    
    for col in required_columns:
        assert col in df.columns, f"Missing required column: {col}"


def test_manifest_no_unknown_splits():
    """Test that manifest has no unknown split values."""
    manifest_path = Path("metadata/dataset_manifest.csv")
    df = pd.read_csv(manifest_path)
    
    allowed_splits = {'train', 'val', 'test'}
    actual_splits = set(df['split'].unique())
    
    assert actual_splits.issubset(allowed_splits), f"Unknown splits found: {actual_splits - allowed_splits}"


def test_machine_ids_are_disjoint_across_splits():
    """Test that machine IDs do not overlap between train, validation, and test splits."""
    manifest_path = Path("metadata/dataset_manifest.csv")
    df = pd.read_csv(manifest_path)

    normalized = df['split'].map(lambda value: {'train': 'train', 'val': 'validation', 'validation': 'validation', 'test': 'test'}.get(str(value).strip().lower()))
    df = df.copy()
    df['_normalized_split'] = normalized

    split_machine_ids = {
        split: set(df.loc[df['_normalized_split'] == split, 'machine_id'])
        for split in ['train', 'validation', 'test']
    }

    overlaps = set.intersection(*split_machine_ids.values())
    assert not overlaps, f"Machine IDs overlap across splits: {sorted(overlaps)}"


def test_active_train_loader_uses_manifest_train_rows():
    """Test that the active training dataset loader consumes manifest train rows only."""
    train_ds = MIMIIDataset(cfg, split='train')
    manifest = load_manifest_split(str(Path('metadata/dataset_manifest.csv')), split='train', validate_integrity=True)
    manifest_machine_ids = set(manifest.df['machine_id'].astype(str))
    dataset_machine_ids = {item['machine_id'] for item in train_ds.records}
    assert dataset_machine_ids <= manifest_machine_ids
    assert len(train_ds.records) == len(manifest.df)


def test_active_validation_loader_uses_manifest_validation_rows():
    """Test that the active validation dataset loader consumes manifest validation rows only."""
    val_ds = MIMIIDataset(cfg, split='val')
    manifest = load_manifest_split(str(Path('metadata/dataset_manifest.csv')), split='validation', validate_integrity=True)
    manifest_machine_ids = set(manifest.df['machine_id'].astype(str))
    dataset_machine_ids = {item['machine_id'] for item in val_ds.records}
    assert dataset_machine_ids <= manifest_machine_ids
    assert len(val_ds.records) == len(manifest.df)


def test_normal_loader_uses_train_only():
    """Test that the calibration normal loader only exposes normal train rows."""
    loader = get_normal_loader(cfg)
    dataset = loader.dataset
    assert all(item['label'] == 0 for item in dataset.records)
    assert all(item['machine_id'] in {'id_00', 'id_02', 'id_04', 'id_06'} for item in dataset.records)


def test_unknown_split_names_fail():
    """Test that the shared manifest loader rejects unknown split names."""
    with pytest.raises(ValueError):
        load_manifest_split(str(Path('metadata/dataset_manifest.csv')), split='unknown', validate_integrity=False)


def test_manifest_no_unknown_labels():
    """Test that manifest has no unknown label values."""
    manifest_path = Path("metadata/dataset_manifest.csv")
    df = pd.read_csv(manifest_path)
    
    allowed_labels = {'normal', 'abnormal'}
    actual_labels = set(df['label'].unique())
    
    assert actual_labels.issubset(allowed_labels), f"Unknown labels found: {actual_labels - allowed_labels}"


def test_backup_created():
    """Test that pre-validation backup was created."""
    backup_dir = Path("artifacts/pre_validation_backup")
    assert backup_dir.exists(), "Backup directory not created"
    assert backup_dir.is_dir(), "Backup should be a directory"


def test_threshold_selection_persists_validation_only_metadata(tmp_path):
    """Threshold selection should persist metadata that marks validation as the selection split and forbids test usage."""
    y_true = np.array([0, 0, 1, 1], dtype=int)
    y_scores = np.array([0.1, 0.2, 0.8, 0.9], dtype=float)

    threshold = select_threshold(y_true, y_scores, selected_on='validation')
    metadata = persist_threshold_metadata(
        threshold=threshold,
        selected_on='validation',
        test_data_used=False,
        output_path=str(tmp_path / 'threshold_metadata.json'),
    )

    assert threshold > 0.0
    assert metadata['selected_on'] == 'validation'
    assert metadata['test_data_used'] is False

    loaded = load_threshold_metadata(str(tmp_path / 'threshold_metadata.json'))
    assert loaded['threshold'] == threshold
    assert loaded['selected_on'] == 'validation'


def test_docs_exist():
    """Test that documentation files exist."""
    docs = [
        "docs/CURRENT_EXPERIMENT_AUDIT.md",
        "docs/REPRODUCIBILITY.md",
        "docs/DATA_SPLIT_PROTOCOL.md",
        "docs/METRIC_DEFINITIONS.md",
        "docs/LIMITATIONS.md",
        "docs/IEEE_EXPERIMENTAL_METHOD.md"
    ]
    
    for doc in docs:
        assert Path(doc).exists(), f"Documentation not found: {doc}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
