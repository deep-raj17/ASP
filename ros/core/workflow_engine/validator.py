"""Structural and semantic workflow validation."""

from __future__ import annotations

from collections import defaultdict, deque

from .types import Severity, ValidationIssue, WorkflowDefinition


def validate_workflow(definition: WorkflowDefinition) -> tuple[ValidationIssue, ...]:
    issues: list[ValidationIssue] = []

    def error(code: str, location: str, message: str, correction: str) -> None:
        issues.append(ValidationIssue(code, Severity.ERROR, location, message, correction))

    if not definition.id.strip():
        error(
            "INVALID_WORKFLOW_DEFINITION",
            "id",
            "Workflow ID is required",
            "Provide a stable non-empty ID",
        )
    ids = [gate.id for gate in definition.gates]
    duplicates = sorted({gate_id for gate_id in ids if ids.count(gate_id) > 1})
    for gate_id in duplicates:
        error(
            "DUPLICATE_IDENTIFIER",
            f"gates.{gate_id}",
            f"Duplicate gate ID: {gate_id}",
            "Use a unique gate ID",
        )
    known = set(ids)
    for gate in definition.gates:
        if not gate.id or not gate.title:
            error(
                "INVALID_WORKFLOW_DEFINITION",
                f"gates.{gate.id or '?'}",
                "Gate ID and title are required",
                "Provide both fields",
            )
        if gate.retry_limit < 0:
            error(
                "INVALID_WORKFLOW_DEFINITION",
                f"gates.{gate.id}.retry_limit",
                "Retry limit cannot be negative",
                "Use zero or a positive integer",
            )
        if gate.allow_waiver and not gate.waiver_policy:
            error(
                "INVALID_WORKFLOW_DEFINITION",
                f"gates.{gate.id}.waiver_policy",
                "Waivable gate requires a policy reference",
                "Declare waiver_policy",
            )
        for prereq in gate.prerequisites:
            if prereq.gate_id not in known:
                error(
                    "INVALID_WORKFLOW_DEFINITION",
                    f"gates.{gate.id}.prerequisites",
                    f"Unknown prerequisite: {prereq.gate_id}",
                    "Reference an existing gate",
                )
            if prereq.gate_id == gate.id:
                error(
                    "CYCLIC_DEPENDENCY",
                    f"gates.{gate.id}.prerequisites",
                    "Gate cannot depend on itself",
                    "Remove the self-reference",
                )
    if definition.gates and not any(gate.entry for gate in definition.gates):
        error(
            "INVALID_WORKFLOW_DEFINITION",
            "gates",
            "At least one entry gate is required",
            "Mark a prerequisite-free gate as entry",
        )
    if definition.gates and not any(gate.terminal for gate in definition.gates):
        error(
            "INVALID_WORKFLOW_DEFINITION",
            "gates",
            "At least one terminal gate is required",
            "Mark a reachable final gate as terminal",
        )
    graph: dict[str, list[str]] = defaultdict(list)
    indegree = {gate_id: 0 for gate_id in known}
    for gate in definition.gates:
        for prereq in gate.prerequisites:
            if prereq.gate_id in known:
                graph[prereq.gate_id].append(gate.id)
                indegree[gate.id] += 1
    queue = deque(sorted(key for key, value in indegree.items() if value == 0))
    visited = []
    while queue:
        current = queue.popleft()
        visited.append(current)
        for child in sorted(graph[current]):
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    if len(visited) != len(known):
        error(
            "CYCLIC_DEPENDENCY",
            "gates",
            "Workflow gate dependency graph contains a cycle",
            "Remove cyclic prerequisite references",
        )
    reachable = set()
    pending = deque(gate.id for gate in definition.gates if gate.entry)
    while pending:
        current = pending.popleft()
        if current in reachable:
            continue
        reachable.add(current)
        pending.extend(graph[current])
    for gate in definition.gates:
        if gate.terminal and gate.id not in reachable:
            error(
                "INVALID_WORKFLOW_DEFINITION",
                f"gates.{gate.id}",
                "Terminal gate is unreachable from any entry gate",
                "Correct prerequisites or entry gates",
            )
    return tuple(issues)
