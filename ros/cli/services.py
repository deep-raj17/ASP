from __future__ import annotations

import json
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from ros.core.registry import RegistryRecord, SQLiteRegistry
from ros.core.workflow_engine.audit import AppendOnlyAuditLog
from ros.core.workflow_engine.engine import WorkflowEngine
from ros.core.workflow_engine.loader import load_workflow
from ros.core.workflow_engine.state_store import JsonStateStore
from ros.core.workflow_engine.types import TransitionRequest


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RosServices:
    """Public service facade used by interfaces."""

    def __init__(self, workspace: str | Path):
        self.workspace = Path(workspace).resolve()
        self._registry = None
        self._workflow = None

    @property
    def registry(self):
        if self._registry is None:
            self._registry = SQLiteRegistry(self.workspace / ".ros" / "registry.db")
        return self._registry

    @property
    def workflow(self):
        if self._workflow is None:
            state = self.workspace / ".ros"
            self._workflow = WorkflowEngine(
                JsonStateStore(state / "workflow-state.json"),
                AppendOnlyAuditLog(state / "workflow-audit.jsonl"),
            )
        return self._workflow

    def init(self, dry_run: bool):
        paths = [self.workspace / ".ros", self.workspace / "projects"]
        existing = [str(path) for path in paths if path.exists()]
        created = [str(path) for path in paths if not path.exists()]
        if not dry_run:
            for path in paths:
                path.mkdir(parents=True, exist_ok=True)
        return {"created": created, "skipped": existing}

    def project_add(self, manifest_path: str, dry_run: bool):
        path = self._safe_path(manifest_path)
        manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
        project_id = manifest["metadata"]["id"]
        record = RegistryRecord(
            "projects", f"project-{project_id}-v1", project_id, "1.0.0", "1.0.0",
            "created", "ros-cli", now(), str(uuid.uuid4()), "cli", "ACTIVE",
            manifest, (), None, f"project-add:{project_id}:1.0.0",
        )
        registry_path = self.workspace / ".ros" / "registry.db"
        if dry_run and not registry_path.exists():
            with tempfile.TemporaryDirectory(prefix="ros-project-preview-") as temporary:
                return SQLiteRegistry(Path(temporary) / "registry.db").append(record, dry_run=True)
        return self.registry.append(record, dry_run=dry_run)

    def project_validate(self, manifest_path: str):
        path = self._safe_path(manifest_path)
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        missing = [key for key in ("api_version", "kind", "metadata", "spec") if key not in raw]
        if missing:
            raise ValueError("SCHEMA_VALIDATION_FAILED")
        return {"valid": True, "missing": [], "path": path.name}

    def project_list(self):
        return self.registry.current_view("projects")

    def project_show(self, project_id: str | None):
        if not project_id:
            raise ValueError("INVALID_ARGUMENTS")
        project = next((item for item in self.project_list() if item["entity_id"] == project_id), None)
        if not project:
            raise KeyError("NOT_FOUND")
        return project

    def project_history(self, project_id: str | None):
        return self.registry.history("projects", project_id)

    def status(self, project_id: str | None):
        projects = self.project_list()
        selected = next((p for p in projects if p["entity_id"] == project_id), None) if project_id else None
        instances = self.workflow.store.load_all()
        workflow = next(iter(instances.values()), None)
        return {
            "project": selected,
            "workflow": workflow.to_dict() if workflow else None,
            "registry_integrity": self.registry.verify_integrity().valid,
            "blocking_reasons": (
                selected["payload"].get("spec", {}).get("imported_state", {}).get("reason_code")
                if selected else None
            ),
        }

    def verify_registry(self):
        report = self.registry.verify_integrity()
        return {
            "valid": report.valid, "record_count": report.record_count,
            "head_checksum": report.head_checksum,
            "issues": [issue.__dict__ for issue in report.issues],
        }

    def workflow_validate(self, path: str):
        definition = load_workflow(self._safe_path(path))
        return {"valid": True, "id": definition.id, "version": str(definition.version)}

    def run_workflow(self, path: str, instance_id: str, dry_run: bool):
        definition = load_workflow(self._safe_path(path))
        state_path = self.workspace / ".ros" / "workflow-state.json"
        if dry_run and not state_path.exists():
            with tempfile.TemporaryDirectory(prefix="ros-workflow-preview-") as temporary:
                root = Path(temporary)
                engine = WorkflowEngine(
                    JsonStateStore(root / "workflow-state.json"),
                    AppendOnlyAuditLog(root / "workflow-audit.jsonl"),
                )
                instance = engine.create_instance(definition, instance_id)
                return {"instance": instance.to_dict(), "planned": True, "next_action": "start"}
        try:
            instance = self.workflow.store.get(instance_id)
            self.workflow.register_definition(definition)
        except Exception:
            instance = self.workflow.create_instance(definition, instance_id, dry_run=dry_run)
            if dry_run:
                return {"instance": instance.to_dict(), "planned": True, "next_action": "start"}
        request = TransitionRequest(
            instance_id, "start", "ros-cli", "human", str(uuid.uuid4()),
            f"start:{instance_id}", instance.revision, dry_run,
        )
        result = self.workflow.transition(request)
        return result.__dict__

    def gate_requirements(self, path: str):
        source = self._safe_path(path)
        definition = load_workflow(source)
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
        raw_gates = {
            str(item.get("id")): item
            for item in raw.get("gates", [])
            if isinstance(item, dict)
        }
        return {
            gate.id: {
                "prerequisites": [item.gate_id for item in gate.prerequisites],
                "waiver_policy": gate.waiver_policy,
                "administrative": gate.administrative,
                "evidence_requirements": raw_gates.get(gate.id, {}).get("requirements", []),
            }
            for gate in definition.gates
        }

    def registry_export(self, path: str, dry_run: bool):
        if dry_run and not (self.workspace / ".ros" / "registry.db").exists():
            with tempfile.TemporaryDirectory(prefix="ros-registry-preview-") as temporary:
                return SQLiteRegistry(Path(temporary) / "registry.db").export_bundle(
                    self._safe_output(path), dry_run=True
                )
        return self.registry.export_bundle(self._safe_output(path), dry_run=dry_run)

    def registry_import(self, path: str, dry_run: bool):
        if dry_run and not (self.workspace / ".ros" / "registry.db").exists():
            with tempfile.TemporaryDirectory(prefix="ros-registry-preview-") as temporary:
                return SQLiteRegistry(Path(temporary) / "registry.db").import_bundle(
                    self._safe_path(path), dry_run=True
                )
        return self.registry.import_bundle(self._safe_path(path), dry_run=dry_run)

    def archive(self, project_id: str, dry_run: bool, approval: str | None):
        if not approval:
            raise PermissionError("APPROVAL_REQUIRED")
        current = next((p for p in self.project_list() if p["entity_id"] == project_id), None)
        if not current:
            raise KeyError("NOT_FOUND")
        record = RegistryRecord(
            "projects", f"project-{project_id}-archive-{uuid.uuid4().hex[:8]}",
            project_id, f"archive-{now()}", "1.0.0", "tombstone", "ros-cli", now(),
            str(uuid.uuid4()), approval, "ARCHIVED", {"approval": approval}, (),
            current["record_id"], f"archive:{project_id}:{approval}",
        )
        return self.registry.append(record, dry_run=dry_run)

    def _safe_path(self, value: str) -> Path:
        path = Path(value)
        if not path.is_absolute():
            path = self.workspace / path
        return path.resolve()

    def _safe_output(self, value: str) -> Path:
        path = self._safe_path(value)
        if self.workspace not in path.parents and path != self.workspace:
            raise ValueError("UNSAFE_OUTPUT_PATH")
        return path
