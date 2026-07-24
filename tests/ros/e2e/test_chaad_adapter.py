import hashlib
import importlib
from pathlib import Path

import yaml

from ros.cli.services import RosServices
from ros.core.workflow_engine.loader import load_workflow


ROOT = Path(__file__).resolve().parents[3]
run = importlib.import_module("projects.chaad.import.import_chaad").run


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_chaad_adapter_replay_is_idempotent_and_non_destructive(tmp_path):
    definition_path = ROOT / "ros/specs/workflows/pmps-1.0.0.yaml"
    definition = load_workflow(definition_path)
    raw = yaml.safe_load(definition_path.read_text(encoding="utf-8"))
    assert definition.id == "PMPS" and str(definition.version) == "1.0.0"
    assert len(raw["gates"][0]["requirements"]) == 13
    requirements = RosServices(tmp_path).gate_requirements(str(definition_path))
    assert len(requirements["PMPS-01"]["evidence_requirements"]) == 13

    protected = [
        ROOT / "metadata/dataset_manifest.csv",
        ROOT / "artifacts/EXP-CHAAD-001/validation_predictions.csv",
        ROOT / "artifacts/EXP-CHAAD-001/validation_predictions_corrected.csv",
    ]
    before = {path: digest(path) for path in protected}

    service = RosServices(tmp_path)
    project = service.project_add(str(ROOT / "projects/chaad/project.yaml"), dry_run=False)
    assert project["record_id"] == "project-chaad-v1"
    first = run(ROOT, tmp_path, dry_run=False)
    evidence_lines = (tmp_path / ".ros/evidence.jsonl").read_text(encoding="utf-8").splitlines()
    audit_lines = (tmp_path / ".ros/workflow-audit.jsonl").read_text(encoding="utf-8").splitlines()
    second = run(ROOT, tmp_path, dry_run=False)

    assert first["gate_state"] == second["gate_state"] == "BLOCKED"
    assert first["workflow_state"] == second["workflow_state"] == "BLOCKED"
    assert first["registry_integrity"]["valid"]
    assert first["registry_integrity"]["record_count"] == 10
    assert all(item["idempotent"] for item in second["registry"])
    assert len((tmp_path / ".ros/evidence.jsonl").read_text(encoding="utf-8").splitlines()) == len(evidence_lines)
    assert len((tmp_path / ".ros/workflow-audit.jsonl").read_text(encoding="utf-8").splitlines()) == len(audit_lines)
    assert service.registry.get("chaad-validation-invalid-v1")["status"] == "INVALID"
    assert {path: digest(path) for path in protected} == before


def test_public_adapter_files_do_not_embed_private_paths():
    for path in (ROOT / "projects/chaad").rglob("*"):
        if path.is_file() and path.suffix in {".yaml", ".md", ".example"}:
            text = path.read_text(encoding="utf-8")
            assert "C:\\ASP\\ASP" not in text
            assert "E:\\MIMII" not in text
