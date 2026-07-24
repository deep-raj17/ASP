"""Safe declarative workflow loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .errors import ErrorCode, WorkflowError
from .types import GateDefinition, GateState, Prerequisite, WorkflowDefinition, WorkflowVersion
from .validator import validate_workflow

SUPPORTED_SCHEMA_VERSIONS = {"ros.workflow/v1"}


def _require_mapping(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise WorkflowError(
            ErrorCode.INVALID_WORKFLOW_DEFINITION,
            f"Expected mapping at {location}",
            location,
        )
    return value


def load_workflow(path: str | Path) -> WorkflowDefinition:
    source = Path(path)
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise WorkflowError(
            ErrorCode.INVALID_WORKFLOW_DEFINITION, str(exc), str(source)
        ) from exc
    root = _require_mapping(raw, "$")
    schema_version = str(root.get("schema_version", ""))
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise WorkflowError(
            ErrorCode.UNSUPPORTED_SCHEMA_VERSION,
            f"Unsupported workflow schema: {schema_version}",
            "schema_version",
        )
    gates = []
    for index, value in enumerate(root.get("gates", [])):
        item = _require_mapping(value, f"gates[{index}]")
        prerequisites = []
        for prereq in item.get("prerequisites", []):
            if isinstance(prereq, str):
                prerequisites.append(Prerequisite(prereq))
            else:
                mapping = _require_mapping(prereq, f"gates[{index}].prerequisites")
                accepted = tuple(
                    GateState(state)
                    for state in mapping.get("accepted_states", ["SATISFIED", "WAIVED"])
                )
                prerequisites.append(Prerequisite(str(mapping["gate_id"]), accepted))
        gates.append(
            GateDefinition(
                id=str(item.get("id", "")),
                title=str(item.get("title", "")),
                prerequisites=tuple(prerequisites),
                entry=bool(item.get("entry", False)),
                terminal=bool(item.get("terminal", False)),
                allow_waiver=bool(item.get("allow_waiver", False)),
                waiver_policy=item.get("waiver_policy"),
                administrative=bool(item.get("administrative", False)),
                parallel_group=item.get("parallel_group"),
                retry_limit=int(item.get("retry_limit", 0)),
            )
        )
    try:
        version = WorkflowVersion.parse(str(root.get("version", "")))
    except ValueError as exc:
        raise WorkflowError(
            ErrorCode.INVALID_WORKFLOW_DEFINITION, str(exc), "version"
        ) from exc
    definition = WorkflowDefinition(
        id=str(root.get("id", "")),
        version=version,
        schema_version=schema_version,
        gates=tuple(gates),
        source=str(source.resolve()),
        stop_conditions=tuple(str(value) for value in root.get("stop_conditions", [])),
    )
    issues = validate_workflow(definition)
    errors = [issue for issue in issues if issue.severity.value == "ERROR"]
    if errors:
        first = errors[0]
        code = (
            ErrorCode(first.error_code)
            if first.error_code in ErrorCode._value2member_map_
            else ErrorCode.INVALID_WORKFLOW_DEFINITION
        )
        raise WorkflowError(code, first.message, first.location)
    return definition
