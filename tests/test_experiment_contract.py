import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from config import Config
from utils.experiment_contract import (
    PROTOCOL_ID,
    assert_split_access,
    canonical_json_hash,
    load_frozen_protocol,
    serialize_config,
    validate_frozen_protocol,
    validate_prediction_export,
    write_immutable_run_contract,
)
from scripts.run_baselines import baseline_global_learned_weights_oof
from scripts.run_reliability_cv import run_grouped_reliability_cv


def test_frozen_protocol_is_valid_and_test_protected():
    protocol = load_frozen_protocol()

    assert protocol["protocol_id"] == PROTOCOL_ID
    assert protocol["dataset"]["test_access_before_phase_8"] == "forbidden"
    assert len(protocol["seeds"]) == 3


@pytest.mark.parametrize("phase", range(1, 8))
def test_development_phases_reject_protected_test(phase: int):
    with pytest.raises(PermissionError, match="forbidden"):
        assert_split_access(phase=phase, split="test")


def test_phase_8_requires_valid_explicit_authorization(tmp_path: Path):
    authorization = tmp_path / "authorization.json"
    authorization.write_text(
        json.dumps({"protocol_id": PROTOCOL_ID, "authorized": True}),
        encoding="utf-8",
    )

    assert (
        assert_split_access(phase=8, split="test", authorization_file=authorization)
        == "test"
    )

    authorization.write_text(
        json.dumps({"protocol_id": PROTOCOL_ID, "authorized": False}),
        encoding="utf-8",
    )
    with pytest.raises(PermissionError, match="Invalid"):
        assert_split_access(phase=8, split="test", authorization_file=authorization)


def test_protocol_validation_rejects_machine_overlap():
    protocol = load_frozen_protocol()
    protocol["dataset"]["split_policy"]["protected_test"]["machine_ids"] = ["id_04"]

    with pytest.raises(ValueError, match="overlap"):
        validate_frozen_protocol(protocol)


def test_config_serialization_is_complete_and_stable():
    serialized = serialize_config(Config())

    assert set(serialized) == {"data", "model", "training", "inference"}
    assert serialized["training"]["random_seed"] == 42
    assert canonical_json_hash(serialized) == canonical_json_hash(serialize_config(Config()))


def test_run_contract_is_create_only(tmp_path: Path):
    output = tmp_path / "run_contract.json"
    contract = {"protocol_id": PROTOCOL_ID, "phase": 4, "split": "validation"}

    digest = write_immutable_run_contract(output, contract)

    assert len(digest) == 64
    with pytest.raises(FileExistsError):
        write_immutable_run_contract(output, contract)


def test_prediction_export_validation_checks_identity_coverage_and_finiteness():
    frame = pd.DataFrame(
        {
            "sample_id": ["a", "b"],
            "true_label": [0, 1],
            "predicted_score": [0.1, 0.9],
            "split": ["val", "val"],
        }
    )

    report = validate_prediction_export(
        frame,
        expected_ids={"a", "b"},
        expected_split="validation",
    )

    assert report["status"] == "PASS"
    assert report["unique_ids"] == 2

    frame.loc[1, "sample_id"] = "a"
    with pytest.raises(ValueError, match="duplicate"):
        validate_prediction_export(
            frame,
            expected_ids={"a", "b"},
            expected_split="validation",
        )


def test_baseline_and_ablation_matrices_match_frozen_protocol():
    protocol = load_frozen_protocol()
    baselines = pd.read_csv("reports/submission_recovery/phase_2/BASELINE_MATRIX.csv")
    ablations = pd.read_csv("reports/submission_recovery/phase_2/ABLATION_MATRIX.csv")

    assert set(protocol["comparators"]) == {
        "reconstruction_only",
        "embedding_distance_only",
        "mahalanobis_only",
        "contrastive_only",
        "equal_weight_fusion",
        "validation_optimized_global_fusion",
        "unconditioned_learned_fusion",
        "reliability_aware_fusion",
        "simple_neural_classifier",
    }
    assert len(baselines) == 9
    assert len(ablations) == 10
    assert not baselines["baseline_id"].duplicated().any()
    assert not ablations["ablation_id"].duplicated().any()


def test_global_fusion_uses_complete_out_of_fold_predictions():
    rows = []
    for index in range(40):
        label = index % 2
        rows.append(
            {
                "recon_error": 0.2 + 0.5 * label + index * 0.001,
                "embed_dist": 0.1 + 0.3 * label + index * 0.001,
                "mahal_dist": 0.4 + 0.2 * label + index * 0.001,
                "contra_dist": 0.3 + 0.1 * label + index * 0.001,
                "label": label,
                "machine_type": "fan" if index % 4 < 2 else "pump",
                "machine_id": "id_00" if index % 4 < 2 else "id_02",
                "source_recording": f"recording_{index:03d}",
            }
        )

    result = baseline_global_learned_weights_oof(pd.DataFrame(rows), n_splits=5)

    assert 0.0 <= result.roc_auc <= 1.0
    assert result.details is not None
    assert len(result.details["fold_weights"]) == 5


def test_reliability_gate_produces_complete_grouped_oof_predictions():
    rows = []
    for index in range(48):
        label = index % 2
        rows.append(
            {
                "sample_id": f"sample_{index:03d}",
                "label": label,
                "split": "validation",
                "machine_type": "fan" if index % 4 < 2 else "pump",
                "machine_id": "id_00" if index % 4 < 2 else "id_02",
                "noise_condition": "0_dB" if index % 3 else "6_dB",
                "source_recording": f"recording_{index:03d}",
                "recon_error": 0.2 + 0.4 * label + index * 0.001,
                "embed_dist": 0.1 + 0.3 * label + index * 0.001,
                "mahal_dist": 0.3 + 0.2 * label + index * 0.001,
                "contra_dist": 0.4 + 0.1 * label + index * 0.001,
                "embedding_0": float(label),
                "embedding_1": float(index % 3) / 3,
                "embedding_2": float(index % 5) / 5,
                "embedding_3": float(index % 7) / 7,
            }
        )

    predictions, report = run_grouped_reliability_cv(
        pd.DataFrame(rows),
        seed=42,
        outer_splits=2,
        inner_splits=2,
        epochs=2,
        patience=1,
        batch_size=16,
        device="cpu",
    )

    assert report["status"] == "PASS"
    assert len(predictions) == len(rows)
    assert predictions["sample_id"].nunique() == len(rows)
    assert predictions["predicted_score"].notna().all()
    assert set(predictions["fold"]) == {0, 1}
