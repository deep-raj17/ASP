"""Read-only execution planning."""

from __future__ import annotations

from collections import defaultdict

from .types import ExecutionPlan, GateState, WorkflowDefinition, WorkflowInstance


def build_execution_plan(
    definition: WorkflowDefinition, instance: WorkflowInstance
) -> ExecutionPlan:
    ready, blocked, failed, completed = [], [], [], []
    unmet: dict[str, tuple[str, ...]] = {}
    groups: dict[str, list[str]] = defaultdict(list)
    approvals = []
    for gate in definition.gates:
        current = instance.gates[gate.id]
        missing = tuple(
            prerequisite.gate_id
            for prerequisite in gate.prerequisites
            if instance.gates[prerequisite.gate_id].state
            not in prerequisite.accepted_states
        )
        if current.state in {GateState.SATISFIED, GateState.WAIVED}:
            completed.append(gate.id)
        elif current.state is GateState.UNSATISFIED:
            failed.append(gate.id)
        elif current.state is GateState.BLOCKED or missing:
            blocked.append(gate.id)
            unmet[gate.id] = missing
        else:
            ready.append(gate.id)
            if gate.parallel_group:
                groups[gate.parallel_group].append(gate.id)
        if gate.allow_waiver and current.state not in {
            GateState.SATISFIED,
            GateState.WAIVED,
        }:
            approvals.append(gate.waiver_policy or gate.id)
    actions = []
    if instance.state.value == "NOT_STARTED":
        actions.append("start")
    actions.extend(f"evaluate:{gate_id}" for gate_id in ready)
    if instance.state.value == "BLOCKED":
        actions.append("resume")
    return ExecutionPlan(
        workflow_instance_id=instance.id,
        workflow_state=instance.state,
        ready_gates=tuple(sorted(ready)),
        blocked_gates=tuple(sorted(blocked)),
        failed_gates=tuple(sorted(failed)),
        completed_gates=tuple(sorted(completed)),
        unmet_prerequisites=unmet,
        parallelizable=tuple(
            tuple(sorted(values))
            for _, values in sorted(groups.items())
            if len(values) > 1
        ),
        next_actions=tuple(actions),
        required_approvals=tuple(sorted(set(approvals))),
    )
