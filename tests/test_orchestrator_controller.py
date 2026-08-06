from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from orchestrator import controller


def valid_decision(**overrides):
    decision = {
        "action_type": "CODEX_TASK",
        "phase": "PHASE 3",
        "title": "Add an offline guard",
        "objective": "Add one fail-closed validation guard.",
        "instructions": ["Implement the guard.", "Add a regression test."],
        "acceptance_criteria": ["The focused regression test passes."],
        "validation_commands": ["python -m pytest tests/example.py -q"],
        "forbidden_actions": ["Do not access the protected test split."],
        "risk_flags": [],
        "reason": "This is a bounded infrastructure repair.",
        "human_decision_required": "",
    }
    decision.update(overrides)
    return decision


def complete_report() -> str:
    headings = (
        "Objective",
        "Files inspected",
        "Files changed",
        "Commands executed",
        "Tests executed",
        "Results",
        "Failures",
        "Evidence paths",
        "Acceptance-criteria assessment",
        "Remaining risks",
        "Recommended next action",
    )
    return "\n\n".join(f"## {heading}\n\nRecorded." for heading in headings)


def test_extract_json_object_accepts_fenced_json():
    payload = valid_decision()
    wrapped = f"```json\n{json.dumps(payload)}\n```"
    assert controller.extract_json_object(wrapped) == payload


def test_validate_decision_rejects_missing_field():
    decision = valid_decision()
    del decision["risk_flags"]
    with pytest.raises(ValueError, match="missing"):
        controller.validate_decision(decision)


def test_validate_decision_rejects_unknown_risk_flag():
    decision = valid_decision(risk_flags=["invented_permission"])
    with pytest.raises(ValueError, match="Unsupported risk"):
        controller.validate_decision(decision)


def test_protected_test_requires_state_and_scoped_approval(tmp_path):
    approval = tmp_path / "PROTECTED_TEST_APPROVAL.md"
    decision = valid_decision(risk_flags=["protected_test_access"])
    state = controller.default_state()

    with patch.object(controller, "PROTECTED_TEST_APPROVAL_FILE", approval):
        gated = controller.apply_human_gates(decision, state)
        assert gated["action_type"] == "HUMAN_APPROVAL"

        state["protected_test_authorised"] = True
        approval.write_text(
            "DECISION: APPROVED\nSCOPE: PROTECTED_TEST\n",
            encoding="utf-8",
        )
        allowed = controller.apply_human_gates(decision, state)
        assert allowed["action_type"] == "CODEX_TASK"


def test_scientific_experiment_requires_reentry():
    decision = valid_decision(risk_flags=["scientific_experiment"])
    state = controller.default_state()
    gated = controller.apply_human_gates(decision, state)
    assert gated["action_type"] == "HUMAN_APPROVAL"
    assert "research re-entry" in gated["human_decision_required"]

    state["research_reentry_authorised"] = True
    assert (
        controller.apply_human_gates(decision, state)["action_type"]
        == "CODEX_TASK"
    )


def test_non_test_high_risk_task_always_stops():
    decision = valid_decision(risk_flags=["git_push_or_merge"])
    gated = controller.apply_human_gates(decision, controller.default_state())
    assert gated["action_type"] == "HUMAN_APPROVAL"
    assert "git_push_or_merge" in gated["human_decision_required"]


def test_scrub_codex_environment_removes_planner_key():
    source = {
        "OPENAI_API_KEY": "planner-secret",
        "CODEX_API_KEY": "codex-secret",
        "GH_TOKEN": "github-secret",
        "DATABASE_PASSWORD": "database-secret",
        "PATH": "safe-path",
        "CODEX_HOME": "safe-home",
    }
    result = controller.scrub_codex_environment(source)
    assert "OPENAI_API_KEY" not in result
    assert "CODEX_API_KEY" not in result
    assert "GH_TOKEN" not in result
    assert "DATABASE_PASSWORD" not in result
    assert result["PATH"] == "safe-path"
    assert result["CODEX_HOME"] == "safe-home"


def test_validate_codex_report_requires_update_and_sections(tmp_path):
    report = tmp_path / "LATEST_CODEX_REPORT.md"
    report.write_text(complete_report(), encoding="utf-8")

    ok, digest = controller.validate_codex_report(report)
    assert ok is True
    assert digest == controller.sha256_file(report)

    ok, reason = controller.validate_codex_report(
        report,
        previous_hash=digest,
    )
    assert ok is False
    assert "not updated" in reason


def test_validate_codex_report_rejects_incomplete_report(tmp_path):
    report = tmp_path / "LATEST_CODEX_REPORT.md"
    report.write_text("## Objective\n\nOnly one section.", encoding="utf-8")
    ok, reason = controller.validate_codex_report(report)
    assert ok is False
    assert "missing sections" in reason


def test_save_json_atomic_round_trips(tmp_path):
    path = tmp_path / "state.json"
    controller.save_json_atomic(path, {"status": "SAFE", "history": []})
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "status": "SAFE",
        "history": [],
    }
    assert not Path(str(path) + ".tmp").exists()


def test_codex_prompt_contains_report_and_protected_path():
    state = controller.default_state()
    prompt = controller.codex_prompt(valid_decision(), state)
    assert "reports/autonomous_loop/LATEST_CODEX_REPORT.md" in prompt
    assert r"E:\MIMII" in prompt
    assert "Never push, merge, publish" in prompt
