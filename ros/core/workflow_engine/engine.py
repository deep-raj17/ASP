"""Deterministic workflow orchestration service."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import replace
from typing import Dict

from .audit import AppendOnlyAuditLog
from .errors import ErrorCode, WorkflowError
from .evaluator import validate_gate_evaluation
from .execution import build_execution_plan
from .state_store import JsonStateStore
from .transitions import is_workflow_transition_allowed
from .types import (
    AuditEvent,
    EvaluationResult,
    ExecutionPlan,
    GateInstance,
    GateState,
    TransitionRequest,
    TransitionResult,
    WorkflowDefinition,
    WorkflowInstance,
    WorkflowState,
    utc_now,
)


class WorkflowEngine:
    def __init__(self, store: JsonStateStore, audit: AppendOnlyAuditLog):
        self.store = store
        self.audit = audit
        self._definitions: Dict[str, WorkflowDefinition] = {}
        self._idempotency: Dict[str, TransitionResult] = {}

    def register_definition(self, definition: WorkflowDefinition) -> None:
        self._definitions[f"{definition.id}@{definition.version}"] = definition

    def create_instance(
        self,
        definition: WorkflowDefinition,
        instance_id: str,
        *,
        dry_run: bool = False,
    ) -> WorkflowInstance:
        self.register_definition(definition)
        instance = WorkflowInstance(
            id=instance_id,
            definition_id=definition.id,
            workflow_version=str(definition.version),
            state=WorkflowState.NOT_STARTED,
            gates={gate.id: GateInstance(gate.id) for gate in definition.gates},
            revision=1,
        )
        self.store.save(instance, expected_revision=0, dry_run=dry_run)
        return instance

    def plan(self, instance_id: str) -> ExecutionPlan:
        instance = self.store.get(instance_id)
        return build_execution_plan(self._definition(instance), instance)

    def transition(self, request: TransitionRequest) -> TransitionResult:
        existing = self._idempotency.get(request.idempotency_key)
        if existing:
            return replace(existing, idempotent=True)
        instance = self.store.get(request.workflow_instance_id)
        if instance.revision != request.expected_revision:
            raise WorkflowError(
                ErrorCode.CONCURRENCY_CONFLICT,
                f"Expected revision {request.expected_revision}, found {instance.revision}",
            )
        definition = self._definition(instance)
        if request.action == "start":
            result, updated = self._workflow_change(
                instance, request, WorkflowState.READY, "Workflow validated and ready"
            )
        elif request.action == "run":
            result, updated = self._workflow_change(
                instance, request, WorkflowState.RUNNING, "Workflow execution requested"
            )
        elif request.action == "cancel":
            if not request.reason.strip():
                raise WorkflowError(
                    ErrorCode.INVALID_TRANSITION, "Cancellation requires a reason"
                )
            result, updated = self._workflow_change(
                instance, request, WorkflowState.CANCELLED, request.reason
            )
        elif request.action == "resume":
            if instance.state not in {WorkflowState.BLOCKED, WorkflowState.FAILED}:
                raise WorkflowError(
                    ErrorCode.INVALID_TRANSITION,
                    f"Cannot resume workflow in {instance.state.value}",
                )
            result, updated = self._workflow_change(
                instance, request, WorkflowState.READY, "Resume prerequisites rechecked"
            )
        elif request.action == "evaluate_gate":
            result, updated = self._evaluate_gate(definition, instance, request)
        else:
            raise WorkflowError(
                ErrorCode.INVALID_TRANSITION, f"Unknown action: {request.action}"
            )
        if not request.dry_run:
            self.store.save(updated, instance.revision)
            if result.audit_event_id:
                self.audit.append(self._audit_event(instance, updated, request, result))
            self._idempotency[request.idempotency_key] = result
        return result

    def _evaluate_gate(
        self,
        definition: WorkflowDefinition,
        instance: WorkflowInstance,
        request: TransitionRequest,
    ) -> tuple[TransitionResult, WorkflowInstance]:
        if not request.evaluation:
            raise WorkflowError(ErrorCode.INVALID_TRANSITION, "Evaluation input required")
        gate_def = next(
            (gate for gate in definition.gates if gate.id == request.evaluation.gate_id),
            None,
        )
        if not gate_def:
            raise WorkflowError(
                ErrorCode.INVALID_TRANSITION,
                f"Unknown gate: {request.evaluation.gate_id}",
            )
        validate_gate_evaluation(gate_def, request.evaluation)
        for prereq in gate_def.prerequisites:
            if instance.gates[prereq.gate_id].state not in prereq.accepted_states:
                raise WorkflowError(
                    ErrorCode.PREREQUISITE_NOT_SATISFIED,
                    f"Gate {prereq.gate_id} is not satisfied",
                )
        target = GateState(request.evaluation.result.value)
        previous_gate = instance.gates[gate_def.id]
        if previous_gate.state is target:
            result = self._result(
                previous_gate.state.value,
                target.value,
                target.value,
                False,
                "Identical gate evaluation already applied",
                ErrorCode.IDEMPOTENT_NO_CHANGE.value,
                request,
                None,
            )
            return result, instance
        gates = dict(instance.gates)
        gates[gate_def.id] = GateInstance(
            gate_def.id, target, previous_gate.attempts + 1
        )
        workflow_state = instance.state
        if target is GateState.BLOCKED:
            workflow_state = WorkflowState.BLOCKED
        elif target is GateState.UNSATISFIED:
            workflow_state = WorkflowState.FAILED
        elif all(
            gates[gate.id].state in {GateState.SATISFIED, GateState.WAIVED}
            for gate in definition.gates
        ):
            workflow_state = WorkflowState.COMPLETED
        elif workflow_state in {WorkflowState.READY, WorkflowState.BLOCKED}:
            workflow_state = WorkflowState.RUNNING
        event_id = str(uuid.uuid4())
        updated = replace(
            instance, gates=gates, state=workflow_state, revision=instance.revision + 1
        )
        result = self._result(
            previous_gate.state.value,
            target.value,
            target.value,
            True,
            f"Gate evaluation accepted from {request.evaluation.evaluator_identity}",
            None,
            request,
            event_id,
        )
        return result, updated

    def _workflow_change(
        self,
        instance: WorkflowInstance,
        request: TransitionRequest,
        target: WorkflowState,
        reason: str,
    ) -> tuple[TransitionResult, WorkflowInstance]:
        if not is_workflow_transition_allowed(instance.state, target):
            code = (
                ErrorCode.WORKFLOW_TERMINAL
                if instance.state in {WorkflowState.COMPLETED, WorkflowState.CANCELLED}
                else ErrorCode.INVALID_TRANSITION
            )
            raise WorkflowError(
                code, f"{instance.state.value} cannot transition to {target.value}"
            )
        event_id = str(uuid.uuid4())
        updated = replace(instance, state=target, revision=instance.revision + 1)
        return (
            self._result(
                instance.state.value,
                target.value,
                target.value,
                True,
                reason,
                None,
                request,
                event_id,
            ),
            updated,
        )

    def _definition(self, instance: WorkflowInstance) -> WorkflowDefinition:
        key = f"{instance.definition_id}@{instance.workflow_version}"
        try:
            return self._definitions[key]
        except KeyError as exc:
            raise WorkflowError(
                ErrorCode.INVALID_WORKFLOW_DEFINITION,
                f"Definition not registered: {key}",
            ) from exc

    @staticmethod
    def _result(
        previous: str,
        requested: str,
        final: str,
        mutated: bool,
        reason: str,
        error: str | None,
        request: TransitionRequest,
        event_id: str | None,
    ) -> TransitionResult:
        return TransitionResult(
            previous,
            requested,
            final,
            mutated and not request.dry_run,
            reason,
            error,
            event_id,
            utc_now(),
            request.actor,
            request.correlation_id,
        )

    @staticmethod
    def _request_checksum(request: TransitionRequest) -> str:
        payload = {
            "instance": request.workflow_instance_id,
            "action": request.action,
            "actor": request.actor,
            "correlation": request.correlation_id,
            "idempotency": request.idempotency_key,
            "revision": request.expected_revision,
            "reason": request.reason,
            "evaluation": repr(request.evaluation),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def _audit_event(
        self,
        previous: WorkflowInstance,
        updated: WorkflowInstance,
        request: TransitionRequest,
        result: TransitionResult,
    ) -> AuditEvent:
        evaluation = request.evaluation
        return AuditEvent(
            event_id=result.audit_event_id or str(uuid.uuid4()),
            event_type="GateEvaluated" if evaluation else "WorkflowTransitioned",
            workflow_instance_id=updated.id,
            workflow_version=updated.workflow_version,
            gate_id=evaluation.gate_id if evaluation else None,
            previous_state=result.previous_state,
            new_state=result.final_state,
            actor=request.actor,
            actor_type=request.actor_type,
            timestamp=result.timestamp,
            correlation_id=request.correlation_id,
            request_checksum=self._request_checksum(request),
            evidence_references=evaluation.evidence_references if evaluation else (),
            approval_references=evaluation.approval_references if evaluation else (),
            reason=result.reason,
            outcome="ACCEPTED",
        )
