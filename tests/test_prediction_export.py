import numpy as np
import pandas as pd
import pytest
import torch

from scripts.audit_evaluation_pipeline import (
    assert_prediction_export_valid,
    build_export_validation_report,
    compare_prediction_exports,
    generate_predictions_from_dataset,
)


class TinyPredictionDataset(torch.utils.data.Dataset):
    def __init__(self, count: int = 5):
        self.rows = []
        for idx in range(count):
            self.rows.append(
                {
                    "mel": torch.full((1, 2, 2), float(idx + 1)),
                    "label": torch.tensor(float(idx % 2), dtype=torch.float32),
                    "sample_id": f"val/path/{idx:04d}.wav",
                    "file_path": f"E:/MIMII/val/path/{idx:04d}.wav",
                    "relative_path": f"val/path/{idx:04d}.wav",
                    "split": "val",
                    "machine": "fan",
                    "machine_id": "id_00",
                    "snr": "0_dB",
                }
            )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class DeterministicModel(torch.nn.Module):
    def forward(self, mel):
        logits = mel.flatten(1).mean(dim=1, keepdim=True)
        return {"logits": logits}


def _export(batch_size: int, count: int = 5) -> pd.DataFrame:
    return generate_predictions_from_dataset(
        model=DeterministicModel(),
        dataset=TinyPredictionDataset(count=count),
        device=torch.device("cpu"),
        batch_size=batch_size,
    )


def _valid_report(df: pd.DataFrame, expected_count: int):
    return build_export_validation_report(
        df=df,
        expected_count=expected_count,
        checkpoint_path="checkpoints/best_model.pt",
        checkpoint_sha256="dummy",
    )


def test_short_final_batch_exports_unique_dataset_ids():
    df = _export(batch_size=4, count=5)
    report = _valid_report(df, expected_count=5)

    assert report["exported_prediction_rows"] == 5
    assert report["unique_sample_ids"] == 5
    assert report["duplicate_sample_id_count"] == 0
    assert_prediction_export_valid(report)


def test_batch_size_one_exports_expected_row_count():
    df = _export(batch_size=1, count=5)
    report = _valid_report(df, expected_count=5)

    assert report["exported_prediction_rows"] == 5
    assert report["missing_id_count"] == 0
    assert_prediction_export_valid(report)


def test_different_batch_sizes_have_deterministic_ids_and_scores():
    batch_two = _export(batch_size=2, count=7)
    batch_three = _export(batch_size=3, count=7)
    comparison = compare_prediction_exports(batch_two, batch_three, 2, 3)

    assert comparison["sample_ids_identical_after_sort"] is True
    assert comparison["labels_identical_after_sort"] is True
    assert comparison["scores_equal_within_tolerance"] is True
    assert comparison["status"] == "PASS"


def test_duplicate_detection_fails_loudly():
    df = _export(batch_size=2, count=4)
    df.loc[3, "sample_id"] = df.loc[0, "sample_id"]
    report = _valid_report(df, expected_count=4)

    assert report["duplicate_sample_id_count"] == 1
    with pytest.raises(AssertionError, match="duplicate sample_id"):
        assert_prediction_export_valid(report)


def test_expected_row_count_mismatch_fails_loudly():
    df = _export(batch_size=2, count=4)
    report = _valid_report(df.iloc[:3], expected_count=4)

    with pytest.raises(AssertionError, match="row count"):
        assert_prediction_export_valid(report)


def test_repeated_export_consistency_same_batch_size():
    first = _export(batch_size=2, count=6)
    second = _export(batch_size=2, count=6)

    np.testing.assert_array_equal(first["sample_id"].to_numpy(), second["sample_id"].to_numpy())
    np.testing.assert_array_equal(first["true_label"].to_numpy(), second["true_label"].to_numpy())
    np.testing.assert_allclose(first["predicted_score"].to_numpy(), second["predicted_score"].to_numpy())
