import json
from pathlib import Path

import pytest
import yaml

import ros.cli.main as cli
from ros.cli.main import EXIT, main, parser


def invoke(capsys, *args):
    code = main(list(args))
    output = capsys.readouterr().out
    return code, output


def manifest(path):
    path.write_text(
        yaml.safe_dump(
            {
                "api_version": "ros.dev/v1",
                "kind": "Project",
                "metadata": {"id": "demo"},
                "spec": {"name": "Demo"},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def test_help_and_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    output = capsys.readouterr().out
    assert exc.value.code == 0 and output.strip() == "0.1.0"
    help_page = parser().format_help()
    for command in (
        "init", "project", "status", "verify", "run", "resume", "gate",
        "workflow", "evidence", "registry", "module", "doctor", "export", "archive",
    ):
        assert command in help_page


def test_init_project_status_json_and_idempotency(tmp_path, capsys):
    code, _ = invoke(capsys, "--config", str(tmp_path), "--dry-run", "init")
    assert code == 0 and not (tmp_path / ".ros").exists()
    project = tmp_path / "project.yaml"
    manifest(project)
    code, _ = invoke(
        capsys, "--config", str(tmp_path), "--dry-run", "project", "add", str(project),
    )
    assert code == 0 and not (tmp_path / ".ros").exists()
    code, _ = invoke(capsys, "--config", str(tmp_path), "init")
    assert code == 0
    code, output = invoke(
        capsys, "--config", str(tmp_path), "--format", "json",
        "project", "add", str(project),
    )
    payload = json.loads(output)
    assert code == 0 and payload["success"]
    code, output = invoke(
        capsys, "--config", str(tmp_path), "--format", "json",
        "project", "add", str(project),
    )
    assert json.loads(output)["result"]["idempotent"]
    code, output = invoke(
        capsys, "--config", str(tmp_path), "--project", "demo",
        "--format", "json", "status",
    )
    assert code == 0 and json.loads(output)["result"]["project"]["entity_id"] == "demo"


def test_registry_export_verify_and_archive_approval(tmp_path, capsys):
    invoke(capsys, "--config", str(tmp_path), "init")
    project = tmp_path / "project.yaml"; manifest(project)
    invoke(capsys, "--config", str(tmp_path), "project", "add", str(project))
    code, output = invoke(capsys, "--config", str(tmp_path), "--format", "json", "registry", "verify")
    assert code == 0 and json.loads(output)["result"]["valid"]
    code, _ = invoke(capsys, "--config", str(tmp_path), "--non-interactive", "archive", "demo")
    assert code == EXIT["APPROVAL_REQUIRED"]
    code, _ = invoke(capsys, "--config", str(tmp_path), "--dry-run", "archive", "demo", "--approval", "a1")
    assert code == 0
    assert len(json.loads((tmp_path / ".ros" / "workflow-state.json").read_text()) if (tmp_path / ".ros" / "workflow-state.json").exists() else {}) == 0


def test_workflow_dry_run_and_path_safety(tmp_path, capsys):
    workflow = tmp_path / "workflow.yaml"
    workflow.write_text(
        "schema_version: ros.workflow/v1\nid: demo\nversion: 1.0.0\ngates:\n"
        "  - {id: first, title: First, entry: true, terminal: true}\n",
        encoding="utf-8",
    )
    code, output = invoke(
        capsys, "--config", str(tmp_path), "--dry-run", "--format", "json",
        "run", str(workflow),
    )
    assert code == 0 and json.loads(output)["dry_run"]
    assert not (tmp_path / ".ros").exists()
    invoke(capsys, "--config", str(tmp_path), "init")
    outside = tmp_path.parent / "outside.json"
    code, _ = invoke(capsys, "--config", str(tmp_path), "export", str(outside))
    assert code == EXIT["VALIDATION_FAILURE"]


def test_project_read_commands_validation_and_registry_queries(tmp_path, capsys):
    invoke(capsys, "--config", str(tmp_path), "init")
    project = tmp_path / "project.yaml"; manifest(project)
    invoke(capsys, "--config", str(tmp_path), "project", "add", str(project))
    for command in (
        ("project", "validate", str(project)),
        ("project", "list"),
        ("project", "show", "demo"),
        ("--project", "demo", "project", "history"),
        ("registry", "list"),
        ("registry", "history"),
        ("registry", "show", "project-demo-v1"),
        ("registry", "rebuild-view", "projects"),
        ("doctor",),
    ):
        code, output = invoke(
            capsys, "--config", str(tmp_path), "--format", "json", *command,
        )
        assert code == 0
        assert json.loads(output)["success"]
    bad = tmp_path / "bad.yaml"
    bad.write_text("kind: Project\n", encoding="utf-8")
    code, _ = invoke(capsys, "--config", str(tmp_path), "project", "validate", str(bad))
    assert code == EXIT["VALIDATION_FAILURE"]


def test_workflow_validation_requirements_run_and_stable_envelope(tmp_path, capsys):
    invoke(capsys, "--config", str(tmp_path), "init")
    workflow = tmp_path / "workflow.yaml"
    workflow.write_text(
        "schema_version: ros.workflow/v1\nid: demo\nversion: 1.0.0\ngates:\n"
        "  - {id: first, title: First, entry: true, terminal: true}\n",
        encoding="utf-8",
    )
    for command in (
        ("workflow", "validate", str(workflow)),
        ("gate", "requirements", str(workflow)),
        ("--dry-run", "run", str(workflow)),
    ):
        code, output = invoke(
            capsys, "--config", str(tmp_path), "--format", "json",
            "--correlation-id", "corr-1", *command,
        )
        payload = json.loads(output)
        assert code == 0 and payload["success"]
        assert payload["correlation_id"] == "corr-1"
        assert set(payload) == {
            "command", "command_version", "success", "partial_success", "dry_run",
            "project_id", "workflow_id", "gate_id", "correlation_id", "timestamp",
            "result", "warnings", "errors", "next_actions",
        }
        if command[:2] == ("gate", "requirements"):
            assert payload["result"]["first"]["evidence_requirements"] == []


def test_export_clean_import_noninteractive_and_conflict(tmp_path, capsys):
    source = tmp_path / "source"; target = tmp_path / "target"
    invoke(capsys, "--config", str(source), "init")
    project = source / "project.yaml"; manifest(project)
    invoke(capsys, "--config", str(source), "project", "add", str(project))
    bundle = source / "bundle.json"
    code, _ = invoke(capsys, "--config", str(source), "export", str(bundle))
    assert code == 0 and bundle.exists()
    invoke(capsys, "--config", str(target), "init")
    code, output = invoke(
        capsys, "--config", str(target), "--format", "json", "--non-interactive",
        "registry", "import", str(bundle),
    )
    assert code == 0 and json.loads(output)["dry_run"] is True
    assert json.loads(output)["result"][0]["validated"]
    changed = source / "changed.yaml"
    manifest(changed)
    data = yaml.safe_load(changed.read_text(encoding="utf-8"))
    data["spec"]["name"] = "Changed"
    changed.write_text(yaml.safe_dump(data), encoding="utf-8")
    code, _ = invoke(capsys, "--config", str(source), "project", "add", str(changed))
    assert code == EXIT["VALIDATION_FAILURE"]


def test_deferred_commands_fail_explicitly_and_no_manual_gate_pass(tmp_path, capsys):
    assert "pass" not in next(
        action for action in parser()._actions if action.dest == "command"
    ).choices["gate"]._actions[-2].choices
    for command in (
        ("resume", "instance"),
        ("gate", "show"),
        ("workflow", "list"),
        ("evidence", "list"),
        ("module", "list"),
    ):
        code, output = invoke(
            capsys, "--config", str(tmp_path), "--format", "json", *command,
        )
        payload = json.loads(output)
        assert code == EXIT["VALIDATION_FAILURE"]
        assert payload["errors"][0]["code"] == "COMMAND_NOT_IMPLEMENTED"


def test_secret_masking_and_interrupted_command(tmp_path, capsys, monkeypatch):
    def leak(*_args, **_kwargs):
        raise ValueError("token=top-secret")

    monkeypatch.setattr(cli, "dispatch", leak)
    code, output = invoke(
        capsys, "--config", str(tmp_path), "--format", "json", "doctor",
    )
    assert code == EXIT["VALIDATION_FAILURE"]
    assert "top-secret" not in output and "[REDACTED]" in output

    def interrupt(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "dispatch", interrupt)
    code, output = invoke(
        capsys, "--config", str(tmp_path), "--format", "json", "doctor",
    )
    assert code == 130
    assert json.loads(output)["errors"][0]["code"] == "INTERRUPTED"


def test_invalid_command_is_rejected(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["not-a-command"])
    assert exc.value.code == EXIT["INVALID_ARGUMENTS"]
