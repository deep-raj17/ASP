from pathlib import Path

import pytest
import torch

import paths
from utils.checkpoint import load_model_weights


def _artifact_paths(root: Path) -> paths.ArtifactPaths:
    return paths.ArtifactPaths(
        root=str(root),
        checkpoint_dir=str(root / "checkpoints"),
        artifacts_dir=str(root / "artifacts" / "models"),
        edge_models_dir=str(root / "edge_deploy" / "models"),
        best_model_fp32=str(root / "checkpoints" / "best_model.pt"),
        best_model_fp16=str(root / "artifacts" / "models" / "best_model_fp16.pt"),
        detector_calibration=str(root / "checkpoints" / "detector_calibration.pt"),
        manifest=str(root / "artifacts" / "models" / "manifest.json"),
        classifier_int8_onnx=str(root / "edge_deploy" / "models" / "classifier_int8.onnx"),
        classifier_int8_mirror=str(root / "artifacts" / "onnx_int8" / "classifier_int8.onnx"),
        classifier_fp32_onnx=str(root / "edge_deploy" / "models" / "classifier_fp32.onnx"),
    )


def test_require_inference_artifacts_reports_every_missing_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    root = tmp_path / "repo"
    fake_paths = _artifact_paths(root)
    monkeypatch.setattr(paths, "PATHS", fake_paths)
    monkeypatch.setattr(
        paths,
        "resolve_model_checkpoint",
        lambda prefer_fp16_on_cuda=True: (fake_paths.best_model_fp32, "fp32"),
    )

    with pytest.raises(FileNotFoundError) as exc_info:
        paths.require_inference_artifacts()

    message = str(exc_info.value).replace("\\", "/")
    assert "checkpoints/best_model.pt" in message
    assert "checkpoints/detector_calibration.pt" in message


def test_require_inference_artifacts_returns_existing_inputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    root = tmp_path / "repo"
    fake_paths = _artifact_paths(root)
    checkpoint = Path(fake_paths.best_model_fp32)
    calibration = Path(fake_paths.detector_calibration)
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    calibration.touch()
    monkeypatch.setattr(paths, "PATHS", fake_paths)
    monkeypatch.setattr(
        paths,
        "resolve_model_checkpoint",
        lambda prefer_fp16_on_cuda=True: (fake_paths.best_model_fp32, "fp32"),
    )

    assert paths.require_inference_artifacts() == (str(checkpoint), "fp32")


def test_model_loader_uses_restricted_deserialization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    model = torch.nn.Linear(2, 1)
    observed = {}

    def fake_load(path, *, map_location, weights_only):
        observed.update(path=path, map_location=map_location, weights_only=weights_only)
        return {"model_state_dict": model.state_dict(), "epoch": 3}

    monkeypatch.setattr(torch, "load", fake_load)

    epoch, precision = load_model_weights(model, str(checkpoint), torch.device("cpu"))

    assert epoch == 3
    assert precision == "fp32"
    assert observed["weights_only"] is True
