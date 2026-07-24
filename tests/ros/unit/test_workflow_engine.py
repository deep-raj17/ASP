from __future__ import annotations

from pathlib import Path

import pytest

from ros.core.workflow_engine.audit import AppendOnlyAuditLog
from ros.core.workflow_engine.engine import WorkflowEngine
from ros.core.workflow_engine.errors import ErrorCode, WorkflowError
from ros.core.workflow_engine.loader import load_workflow
from ros.core.workflow_engine.state_store import JsonStateStore
from ros.core.workflow_engine.types import (
    EvaluationResult,
    GateEvaluationInput,
    GateState,
    TransitionRequest,
    WorkflowState,
)


FIXTURE = Path("ros/specs/workflows/research-validation-demo.yaml")


def engine(tmp_path):
    service = WorkflowEngine(
        JsonStateStore(tmp_path / "state.json"),
        AppendOnlyAuditLog(tmp_path / "audit.jsonl"),
    )
    definition = load_workflow(FIXTURE)
    service.create_instance(definition, "run-1")
    return service, definition


def request(instance, action, key, *, dry=False, evaluation=None, reason=""):
    return TransitionRequest(
        "run-1",
        action,
        "tester",
        "human",
        "corr-1",
        key,
        instance.revision,
        dry,
        reason,
        evaluation,
    )


def evaluation(gate, result, *, evidence=("ev-1",), approvals=(), policies=()):
    return GateEvaluationInput(
        gate,
        result,
        evidence,
        "verifier",
        "1.0.0",
        policies,
        "2026-07-24T00:00:00+00:00",
        "a" * 64,
        "corr-1",
        approvals,
    )


def test_load_and_plan_parallel_gates(tmp_path):
    service, definition = engine(tmp_path)
    instance = service.store.get("run-1")
    assert instance.state is WorkflowState.NOT_STARTED
    service.transition(request(instance, "start", "start"))
    instance = service.store.get("run-1")
    service.transition(request(instance, "run", "run"))
    instance = service.store.get("run-1")
    service.transition(
        request(
            instance,
            "evaluate_gate",
            "repo",
            evaluation=evaluation("repository", EvaluationResult.SATISFIED),
        )
    )
    plan = service.plan("run-1")
    assert {"data", "method", "ethics"} <= set(plan.ready_gates)
    assert ("data", "method") in plan.parallelizable
    assert definition.id == "research-validation-demo"


def test_prerequisite_and_evidence_enforcement(tmp_path):
    service, _ = engine(tmp_path)
    instance = service.store.get("run-1")
    with pytest.raises(WorkflowError) as exc:
        service.transition(
            request(
                instance,
                "evaluate_gate",
                "early",
                evaluation=evaluation("data", EvaluationResult.SATISFIED),
            )
        )
    assert exc.value.code is ErrorCode.PREREQUISITE_NOT_SATISFIED
    with pytest.raises(WorkflowError) as exc:
        service.transition(
            request(
                instance,
                "evaluate_gate",
                "missing-evidence",
                evaluation=evaluation(
                    "repository", EvaluationResult.SATISFIED, evidence=()
                ),
            )
        )
    assert exc.value.code is ErrorCode.EVIDENCE_REFERENCE_REQUIRED


def test_waiver_requires_policy_and_approval(tmp_path):
    service, _ = engine(tmp_path)
    instance = service.store.get("run-1")
    service.transition(
        request(
            instance,
            "evaluate_gate",
            "repo",
            evaluation=evaluation("repository", EvaluationResult.SATISFIED),
        )
    )
    instance = service.store.get("run-1")
    with pytest.raises(WorkflowError) as exc:
        service.transition(
            request(
                instance,
                "evaluate_gate",
                "waive-bad",
                evaluation=evaluation(
                    "ethics",
                    EvaluationResult.WAIVED,
                    evidence=(),
                    policies=("policy.ethics-waiver.v1",),
                ),
            )
        )
    assert exc.value.code is ErrorCode.APPROVAL_REQUIRED
    service.transition(
        request(
            instance,
            "evaluate_gate",
            "waive-ok",
            evaluation=evaluation(
                "ethics",
                EvaluationResult.WAIVED,
                evidence=(),
                approvals=("approval-1",),
                policies=("policy.ethics-waiver.v1",),
            ),
        )
    )
    assert service.store.get("run-1").gates["ethics"].state is GateState.WAIVED


def test_dry_run_idempotency_and_audit(tmp_path):
    service, _ = engine(tmp_path)
    instance = service.store.get("run-1")
    result = service.transition(request(instance, "start", "dry", dry=True))
    assert not result.mutated
    assert service.store.get("run-1").state is WorkflowState.NOT_STARTED
    assert service.audit.read_all() == ()
    first = service.transition(request(instance, "start", "start"))
    second = service.transition(request(instance, "start", "start"))
    assert first.audit_event_id == second.audit_event_id
    assert second.idempotent
    assert len(service.audit.read_all()) == 1


def test_concurrency_cancellation_and_terminal_preservation(tmp_path):
    service, _ = engine(tmp_path)
    instance = service.store.get("run-1")
    service.transition(request(instance, "start", "start"))
    with pytest.raises(WorkflowError) as exc:
        service.transition(request(instance, "run", "stale"))
    assert exc.value.code is ErrorCode.CONCURRENCY_CONFLICT
    current = service.store.get("run-1")
    with pytest.raises(WorkflowError):
        service.transition(request(current, "cancel", "cancel"))
    service.transition(request(current, "cancel", "cancel", reason="User request"))
    terminal = service.store.get("run-1")
    with pytest.raises(WorkflowError) as exc:
        service.transition(request(terminal, "run", "after-cancel"))
    assert exc.value.code is ErrorCode.WORKFLOW_TERMINAL


def test_block_failure_retry_and_restart(tmp_path):
    service, definition = engine(tmp_path)
    instance = service.store.get("run-1")
    service.transition(
        request(
            instance,
            "evaluate_gate",
            "block",
            evaluation=evaluation("repository", EvaluationResult.BLOCKED),
        )
    )
    assert service.store.get("run-1").state is WorkflowState.BLOCKED
    blocked = service.store.get("run-1")
    service.transition(request(blocked, "resume", "resume"))
    ready = service.store.get("run-1")
    service.transition(
        request(
            ready,
            "evaluate_gate",
            "fail",
            evaluation=evaluation("repository", EvaluationResult.UNSATISFIED),
        )
    )
    assert service.store.get("run-1").state is WorkflowState.FAILED
    restarted = WorkflowEngine(service.store, service.audit)
    restarted.register_definition(definition)
    assert restarted.store.get("run-1").state is WorkflowState.FAILED
    restarted.transition(
        request(restarted.store.get("run-1"), "resume", "retry")
    )
    assert restarted.store.get("run-1").state is WorkflowState.READY
