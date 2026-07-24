"""Atomic workflow state store with optimistic concurrency."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from threading import Lock
from typing import Dict

from .errors import ErrorCode, WorkflowError
from .types import GateInstance, GateState, WorkflowInstance, WorkflowState


class JsonStateStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def load_all(self) -> Dict[str, WorkflowInstance]:
        if not self.path.exists():
            return {}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise WorkflowError(ErrorCode.STATE_STORE_FAILURE, str(exc)) from exc
        return {key: self._decode(value) for key, value in raw.items()}

    def get(self, instance_id: str) -> WorkflowInstance:
        try:
            return self.load_all()[instance_id]
        except KeyError as exc:
            raise WorkflowError(
                ErrorCode.STATE_STORE_FAILURE, f"Workflow instance not found: {instance_id}"
            ) from exc

    def save(
        self, instance: WorkflowInstance, expected_revision: int, dry_run: bool = False
    ) -> None:
        with self._lock:
            current = self.load_all()
            existing = current.get(instance.id)
            actual = existing.revision if existing else 0
            if actual != expected_revision:
                raise WorkflowError(
                    ErrorCode.CONCURRENCY_CONFLICT,
                    f"Expected revision {expected_revision}, found {actual}",
                )
            if dry_run:
                return
            current[instance.id] = instance
            payload = {
                key: value.to_dict() for key, value in sorted(current.items())
            }
            fd, temp_name = tempfile.mkstemp(
                dir=str(self.path.parent), prefix=self.path.name, suffix=".tmp"
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                    json.dump(payload, handle, sort_keys=True, indent=2)
                    handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temp_name, self.path)
            finally:
                if os.path.exists(temp_name):
                    os.unlink(temp_name)

    @staticmethod
    def _decode(value: dict) -> WorkflowInstance:
        gates = {
            key: GateInstance(
                gate_id=item["gate_id"],
                state=GateState(item["state"]),
                attempts=int(item.get("attempts", 0)),
            )
            for key, item in value["gates"].items()
        }
        return WorkflowInstance(
            id=value["id"],
            definition_id=value["definition_id"],
            workflow_version=value["workflow_version"],
            state=WorkflowState(value["state"]),
            gates=gates,
            revision=int(value["revision"]),
        )
