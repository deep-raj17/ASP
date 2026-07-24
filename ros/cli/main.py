from __future__ import annotations

import argparse
import json
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ros import __version__
from ros.core.workflow_engine.errors import WorkflowError

from .services import RosServices

EXIT = {
    "SUCCESS": 0, "INVALID_ARGUMENTS": 2, "VALIDATION_FAILURE": 3,
    "NOT_FOUND": 4, "WORKFLOW_BLOCKED": 5, "GATE_UNSATISFIED": 6,
    "VERIFICATION_FAILED": 7, "APPROVAL_REQUIRED": 8, "POLICY_VIOLATION": 9,
    "INTEGRITY_FAILURE": 10, "CONCURRENCY_CONFLICT": 11,
    "PARTIAL_SUCCESS": 12, "INTERNAL_ERROR": 20,
}
ERROR_EXIT = {
    "CONCURRENCY_CONFLICT": EXIT["CONCURRENCY_CONFLICT"],
    "NOT_FOUND": EXIT["NOT_FOUND"],
    "RECORD_NOT_FOUND": EXIT["NOT_FOUND"],
    "VERSION_NOT_FOUND": EXIT["NOT_FOUND"],
    "WORKFLOW_BLOCKED": EXIT["WORKFLOW_BLOCKED"],
    "GATE_UNSATISFIED": EXIT["GATE_UNSATISFIED"],
    "CHECKSUM_MISMATCH": EXIT["INTEGRITY_FAILURE"],
    "IMPORT_VALIDATION_FAILED": EXIT["INTEGRITY_FAILURE"],
    "STORAGE_CORRUPTION": EXIT["INTEGRITY_FAILURE"],
}
SECRET = re.compile(
    r"(?i)(bearer\s+)[^\s,;]+|((?:api[_-]?key|token|password|secret)\s*[=:]\s*)[^\s,;]+"
)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="ros")
    root.add_argument("--project")
    root.add_argument("--config", default=".")
    root.add_argument("--format", choices=("text", "json"), default="text")
    root.add_argument("--dry-run", action="store_true")
    root.add_argument("--non-interactive", action="store_true")
    root.add_argument("--verbose", action="store_true")
    root.add_argument("--quiet", action="store_true")
    root.add_argument("--correlation-id")
    root.add_argument("--no-color", action="store_true")
    root.add_argument("--version", action="version", version=__version__)
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("init")
    project = commands.add_parser("project"); ps = project.add_subparsers(dest="subcommand", required=True)
    add = ps.add_parser("add"); add.add_argument("manifest")
    show = ps.add_parser("show"); show.add_argument("id", nargs="?")
    ps.add_parser("list")
    validate = ps.add_parser("validate"); validate.add_argument("manifest")
    ps.add_parser("history")
    for name in ("enable-module", "disable-module"):
        sub = ps.add_parser(name); sub.add_argument("module")
    commands.add_parser("status")
    verify = commands.add_parser("verify"); verify.add_argument("target", choices=("evidence","registry","workflow","project","artifact","all")); verify.add_argument("path", nargs="?")
    run = commands.add_parser("run"); run.add_argument("workflow"); run.add_argument("--instance", default="default")
    resume = commands.add_parser("resume"); resume.add_argument("instance")
    gate = commands.add_parser("gate"); gate.add_argument("subcommand", choices=("show","list","history","evaluate","requirements","explain")); gate.add_argument("workflow", nargs="?")
    workflow = commands.add_parser("workflow"); workflow.add_argument("subcommand", choices=("list","show","validate","start","history","plan","cancel")); workflow.add_argument("path", nargs="?")
    evidence = commands.add_parser("evidence"); evidence.add_argument("subcommand", choices=("add","show","list","verify","lineage","history","supersede","integrity")); evidence.add_argument("value", nargs="?")
    registry = commands.add_parser("registry"); registry.add_argument("subcommand", choices=("list","show","history","verify","export","import","rebuild-view")); registry.add_argument("value", nargs="?")
    module = commands.add_parser("module"); module.add_argument("subcommand", choices=("list","show","validate","compatibility","enable","disable")); module.add_argument("value", nargs="?")
    commands.add_parser("doctor")
    export = commands.add_parser("export"); export.add_argument("path")
    archive = commands.add_parser("archive"); archive.add_argument("project_id"); archive.add_argument("--approval")
    return root


def envelope(args, result=None, errors=None, warnings=None, success=True):
    return {
        "command": args.command, "command_version": "1.0.0", "success": success,
        "partial_success": False, "dry_run": args.dry_run, "project_id": args.project,
        "workflow_id": getattr(args, "workflow", None), "gate_id": None,
        "correlation_id": args.correlation_id or str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(), "result": result,
        "warnings": warnings or [], "errors": errors or [], "next_actions": [],
    }


def dispatch(args, service: RosServices):
    if args.command == "init": return service.init(args.dry_run)
    if args.command == "project":
        if args.subcommand == "add": return service.project_add(args.manifest, args.dry_run)
        if args.subcommand == "validate": return service.project_validate(args.manifest)
        if args.subcommand == "list": return service.project_list()
        if args.subcommand == "show": return service.project_show(args.id)
        if args.subcommand == "history": return service.project_history(args.project)
        return {"module": args.module, "requested": args.subcommand, "historical_workflows_unchanged": True}
    if args.command == "status": return service.status(args.project)
    if args.command == "verify":
        if args.target in {"registry", "all"}: return service.verify_registry()
        if args.target in {"project", "workflow"} and args.path: return service.project_validate(args.path) if args.target == "project" else service.workflow_validate(args.path)
        raise ValueError("INVALID_ARGUMENTS")
    if args.command == "run": return service.run_workflow(args.workflow, args.instance, args.dry_run)
    if args.command == "gate" and args.subcommand == "requirements": return service.gate_requirements(args.workflow)
    if args.command == "workflow" and args.subcommand == "validate": return service.workflow_validate(args.path)
    if args.command == "registry":
        if args.subcommand == "verify": return service.verify_registry()
        if args.subcommand == "export": return service.registry_export(args.value, args.dry_run)
        if args.subcommand == "import":
            args.dry_run = args.dry_run or args.non_interactive
            return service.registry_import(args.value, args.dry_run)
        if args.subcommand == "history": return service.registry.history()
        if args.subcommand == "list": return sorted({item["registry"] for item in service.registry.history()})
        if args.subcommand == "show": return service.registry.get(args.value)
        if args.subcommand == "rebuild-view": return service.registry.rebuild_view(args.value)
    if args.command == "doctor":
        return {"workspace": str(service.workspace), "registry": service.verify_registry(), "python": sys.version.split()[0]}
    if args.command == "export": return service.registry_export(args.path, args.dry_run)
    if args.command == "archive": return service.archive(args.project_id, args.dry_run, args.approval)
    if args.command in {"module", "evidence", "workflow", "gate", "resume"}:
        raise ValueError("COMMAND_NOT_IMPLEMENTED")
    raise ValueError("INVALID_ARGUMENTS")


def render_text(payload):
    if payload["success"]:
        return json.dumps(payload["result"], indent=2, sort_keys=True, default=str)
    return "\n".join(item["message"] for item in payload["errors"])


def redact(value):
    if isinstance(value, str):
        return SECRET.sub(lambda match: (match.group(1) or match.group(2)) + "[REDACTED]", value)
    if isinstance(value, dict):
        return {key: ("[REDACTED]" if key.lower() in {"password", "secret", "token", "api_key"}
                      else redact(item)) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    return value


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    service = RosServices(Path(args.config))
    try:
        result = dispatch(args, service)
        payload, code = envelope(args, result=result), EXIT["SUCCESS"]
    except PermissionError as exc:
        payload, code = envelope(args, errors=[{"code": str(exc), "message": str(exc)}], success=False), EXIT["APPROVAL_REQUIRED"]
    except WorkflowError as exc:
        message = str(exc)
        error_name = getattr(exc.code, "value", str(exc.code))
        payload = envelope(args, errors=[{"code": error_name, "message": message}], success=False)
        code = ERROR_EXIT.get(error_name, EXIT["VALIDATION_FAILURE"])
    except (ValueError, KeyError) as exc:
        message = str(exc).strip("'")
        payload = envelope(args, errors=[{"code": message, "message": message}], success=False)
        code = ERROR_EXIT.get(message, EXIT["VALIDATION_FAILURE"])
    except KeyboardInterrupt:
        payload = envelope(
            args, errors=[{"code": "INTERRUPTED", "message": "Command interrupted"}],
            success=False,
        )
        code = 130
    except Exception as exc:
        payload, code = envelope(args, errors=[{"code": type(exc).__name__, "message": str(exc)}], success=False), EXIT["INTERNAL_ERROR"]
    if not args.quiet:
        safe_payload = redact(payload)
        print(json.dumps(safe_payload, sort_keys=True, default=str) if args.format == "json" else render_text(safe_payload))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
