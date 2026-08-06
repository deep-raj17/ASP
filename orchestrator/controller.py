"""Evidence-preserving planner -> Codex orchestration for CHAAD.

The controller deliberately separates the API planning agent from the local
Codex implementation agent. It stops at human-approval gates and never grants
itself permission to access protected data or perform external publication
actions.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
ORCHESTRATOR_DIR = ROOT / "orchestrator"
RUNTIME_DIR = ORCHESTRATOR_DIR / "runtime"
REPORT_DIR = ROOT / "reports" / "autonomous_loop"

MASTER_GOAL_FILE = ORCHESTRATOR_DIR / "MASTER_GOAL.md"
STATE_FILE = ORCHESTRATOR_DIR / "state.json"
LATEST_REPORT_FILE = REPORT_DIR / "LATEST_CODEX_REPORT.md"
PROTECTED_TEST_APPROVAL_FILE = (
    ROOT / "reports" / "submission_recovery" / "PROTECTED_TEST_APPROVAL.md"
)

PLANNER_MODEL = os.getenv("PLANNER_MODEL", "gpt-5.1")
PROTECTED_PATH_TEXT = r"E:\MIMII"

MAX_REPORT_CHARS = 60_000
MAX_GIT_SNAPSHOT_CHARS = 20_000
DEFAULT_TIMEOUT_SECONDS = 60 * 60
DEFAULT_MAX_ITERATIONS = 20
PLANNER_TIMEOUT_SECONDS = 180

ALLOWED_ACTIONS = {
    "CODEX_TASK",
    "HUMAN_APPROVAL",
    "BLOCKED",
    "COMPLETE",
}
ALLOWED_RISK_FLAGS = {
    "credentials",
    "dataset_download",
    "dataset_write",
    "destructive_operation",
    "frozen_protocol_change",
    "git_push_or_merge",
    "paid_compute",
    "protected_test_access",
    "publication_or_submission",
    "scientific_experiment",
}
REQUIRED_DECISION_FIELDS = {
    "action_type",
    "phase",
    "title",
    "objective",
    "instructions",
    "acceptance_criteria",
    "validation_commands",
    "forbidden_actions",
    "risk_flags",
    "reason",
    "human_decision_required",
}
REQUIRED_REPORT_SECTIONS = (
    "objective",
    "files inspected",
    "files changed",
    "commands executed",
    "tests executed",
    "results",
    "failures",
    "evidence paths",
    "acceptance",
    "remaining risks",
    "recommended next action",
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def ensure_directories() -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_text(path: Path, fallback: str = "") -> str:
    if not path.exists():
        return fallback
    return path.read_text(encoding="utf-8", errors="replace")


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def save_json_atomic(path: Path, data: Mapping[str, Any]) -> None:
    write_text_atomic(
        path,
        json.dumps(dict(data), indent=2, ensure_ascii=False) + "\n",
    )


def default_state() -> dict[str, Any]:
    now = utc_now()
    return {
        "programme": "CHAAD_IEEE_SUBMISSION_RECOVERY",
        "iteration": 0,
        "status": "INITIALISING",
        "current_phase": "",
        "last_action": "",
        "next_action": "",
        "human_approval_required": False,
        "protected_test_authorised": False,
        "research_reentry_authorised": False,
        "stop_reason": "",
        "history": [],
        "created_at": now,
        "updated_at": now,
    }


def load_state() -> dict[str, Any]:
    if not STATE_FILE.exists():
        return default_state()

    data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Controller state must be a JSON object.")

    merged = default_state()
    merged.update(data)
    if not merged.get("created_at"):
        merged["created_at"] = utc_now()
    if not merged.get("updated_at"):
        merged["updated_at"] = merged["created_at"]
    if not isinstance(merged.get("history"), list):
        raise ValueError("Controller state history must be a list.")
    if not isinstance(merged.get("iteration"), int):
        raise ValueError("Controller state iteration must be an integer.")
    return merged


def run_capture(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def verify_repository() -> None:
    result = run_capture(["git", "rev-parse", "--show-toplevel"])
    if result.returncode != 0:
        raise RuntimeError("The controller must run inside a Git repository.")

    git_root = Path(result.stdout.strip()).resolve()
    if git_root != ROOT.resolve():
        raise RuntimeError(
            f"Expected repository root {ROOT}, but Git reports {git_root}."
        )


def repository_is_dirty() -> bool:
    result = run_capture(["git", "status", "--porcelain=v1"])
    if result.returncode != 0:
        raise RuntimeError(f"git status failed: {result.stderr.strip()}")
    return bool(result.stdout.strip())


def git_snapshot() -> str:
    commands = [
        ["git", "status", "--short"],
        ["git", "diff", "--stat"],
        ["git", "diff", "--cached", "--stat"],
        ["git", "rev-parse", "HEAD"],
        ["git", "branch", "--show-current"],
    ]
    sections: list[str] = []

    for command in commands:
        result = run_capture(command)
        sections.append(
            f"$ {' '.join(command)}\n"
            f"exit_code={result.returncode}\n"
            f"{result.stdout}\n"
            f"{result.stderr}"
        )

    return "\n\n".join(sections)


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()

    if cleaned.startswith("```"):
        lines = cleaned.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines)
        if cleaned.lstrip().lower().startswith("json"):
            cleaned = cleaned.lstrip()[4:].lstrip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Planner did not return a usable JSON object.")
        parsed = json.loads(cleaned[start : end + 1])

    if not isinstance(parsed, dict):
        raise ValueError("Planner response must be a JSON object.")
    return parsed


def _require_string(decision: Mapping[str, Any], field: str) -> None:
    value = decision[field]
    if not isinstance(value, str):
        raise ValueError(f"Planner field {field!r} must be a string.")


def _require_string_list(decision: Mapping[str, Any], field: str) -> None:
    value = decision[field]
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError(
            f"Planner field {field!r} must be a list of non-empty strings."
        )


def validate_decision(decision: dict[str, Any]) -> dict[str, Any]:
    missing = REQUIRED_DECISION_FIELDS.difference(decision)
    if missing:
        raise ValueError(f"Planner response is missing: {sorted(missing)}")

    for field in (
        "action_type",
        "phase",
        "title",
        "objective",
        "reason",
        "human_decision_required",
    ):
        _require_string(decision, field)

    for field in (
        "instructions",
        "acceptance_criteria",
        "validation_commands",
        "forbidden_actions",
        "risk_flags",
    ):
        if field == "validation_commands" and decision[field] == []:
            continue
        if field == "risk_flags" and decision[field] == []:
            continue
        _require_string_list(decision, field)

    if decision["action_type"] not in ALLOWED_ACTIONS:
        raise ValueError(
            f"Unsupported action_type: {decision['action_type']}"
        )

    unknown_flags = set(decision["risk_flags"]).difference(ALLOWED_RISK_FLAGS)
    if unknown_flags:
        raise ValueError(f"Unsupported risk flags: {sorted(unknown_flags)}")

    if decision["action_type"] == "CODEX_TASK":
        if not decision["instructions"]:
            raise ValueError("CODEX_TASK requires at least one instruction.")
        if not decision["acceptance_criteria"]:
            raise ValueError(
                "CODEX_TASK requires at least one acceptance criterion."
            )
    return decision


def planning_prompt(
    master_goal: str,
    state: Mapping[str, Any],
    codex_report: str,
    repository_snapshot: str,
) -> str:
    return f"""
You are the scientific project manager and independent reviewer for the
CHAAD IEEE submission recovery programme.

Select exactly one next task for a local Codex coding agent.

MASTER PROGRAMME:
{master_goal}

CURRENT CONTROLLER STATE:
{json.dumps(dict(state), indent=2)}

LATEST CODEX REPORT:
{codex_report[-MAX_REPORT_CHARS:]}

CURRENT GIT SNAPSHOT:
{repository_snapshot[-MAX_GIT_SNAPSHOT_CHARS:]}

RULES:

1. Return exactly one JSON object and no Markdown.
2. Do not invent repository evidence.
3. Select the smallest task that materially advances the current phase.
4. Include measurable acceptance criteria.
5. Do not request protected-test access unless all pre-test gates pass.
6. Never authorise publication, submission, Git push, merge, deletion,
   credential disclosure, paid resources, dataset download/write, or
   modification of {PROTECTED_PATH_TEXT}.
7. Identify every high-risk dependency in risk_flags.
8. Use action_type HUMAN_APPROVAL whenever an identified risk requires a
   human decision.
9. Use COMPLETE only when the evidence-defined programme is complete.
10. Use BLOCKED when no safe local task can advance the programme.
11. Otherwise use CODEX_TASK.
12. Do not return broad instructions such as "continue the project".
13. Require Codex to write its report to
    reports/autonomous_loop/LATEST_CODEX_REPORT.md.

Return this exact structure:

{{
  "action_type": "CODEX_TASK | HUMAN_APPROVAL | BLOCKED | COMPLETE",
  "phase": "current phase",
  "title": "short task title",
  "objective": "one concrete objective",
  "instructions": ["specific instruction"],
  "acceptance_criteria": ["measurable criterion"],
  "validation_commands": ["safe local command, or empty list"],
  "forbidden_actions": ["explicit forbidden action"],
  "risk_flags": [],
  "reason": "why this is the next task",
  "human_decision_required": ""
}}
""".strip()


def ask_planner(
    client: Any,
    master_goal: str,
    state: Mapping[str, Any],
    codex_report: str,
    repository_snapshot: str,
) -> dict[str, Any]:
    response = client.responses.create(
        model=PLANNER_MODEL,
        input=planning_prompt(
            master_goal=master_goal,
            state=state,
            codex_report=codex_report,
            repository_snapshot=repository_snapshot,
        ),
    )
    return validate_decision(extract_json_object(response.output_text))


def protected_test_approval_is_valid(state: Mapping[str, Any]) -> bool:
    if state.get("protected_test_authorised") is not True:
        return False
    if not PROTECTED_TEST_APPROVAL_FILE.is_file():
        return False

    approval = load_text(PROTECTED_TEST_APPROVAL_FILE).upper()
    return (
        "DECISION: APPROVED" in approval
        and "SCOPE: PROTECTED_TEST" in approval
    )


def apply_human_gates(
    decision: dict[str, Any],
    state: Mapping[str, Any],
) -> dict[str, Any]:
    if decision["action_type"] != "CODEX_TASK":
        return decision

    risk_flags = set(decision["risk_flags"])
    unmet: list[str] = []

    if "protected_test_access" in risk_flags:
        if not protected_test_approval_is_valid(state):
            unmet.append(
                "Protected-test access requires both the state flag and a "
                "scoped approval record."
            )
        else:
            risk_flags.remove("protected_test_access")

    if "scientific_experiment" in risk_flags:
        if state.get("research_reentry_authorised") is not True:
            unmet.append(
                "Scientific experiments require explicit research re-entry."
            )
        else:
            risk_flags.remove("scientific_experiment")

    if risk_flags:
        unmet.append(
            "Human approval is required for risk flags: "
            + ", ".join(sorted(risk_flags))
        )

    if not unmet:
        return decision

    gated = dict(decision)
    gated["action_type"] = "HUMAN_APPROVAL"
    gated["human_decision_required"] = " ".join(unmet)
    gated["reason"] = (
        f"{decision['reason']} Controller safety gate stopped execution."
    )
    return gated


def codex_prompt(
    decision: Mapping[str, Any],
    state: Mapping[str, Any],
) -> str:
    def bullets(field: str) -> str:
        values = decision.get(field, [])
        return "\n".join(f"- {item}" for item in values)

    return f"""
You are the implementation agent operating in the CHAAD repository.

Read and obey:

- AGENTS.md
- orchestrator/MASTER_GOAL.md
- orchestrator/state.json
- relevant current repository documentation and lower-level evidence

CURRENT PHASE:
{decision["phase"]}

TASK:
{decision["title"]}

OBJECTIVE:
{decision["objective"]}

IMPLEMENTATION INSTRUCTIONS:
{bullets("instructions")}

ACCEPTANCE CRITERIA:
{bullets("acceptance_criteria")}

SUGGESTED VALIDATION COMMANDS:
{bullets("validation_commands") or "- Determine safe repository-specific commands."}

FORBIDDEN ACTIONS:
{bullets("forbidden_actions")}

GLOBAL RESTRICTIONS:

- Never modify {PROTECTED_PATH_TEXT}.
- Never access protected test data unless state and the approval record both
  explicitly authorize it.
- Never fabricate results or references.
- Never push, merge, publish, submit, upload datasets, or purchase resources.
- Never expose credentials.
- Never overwrite or delete historical evidence.
- Work only inside the repository.
- Preserve unrelated dirty-worktree changes.
- Inspect before editing and implement the smallest correct change.
- Run only safe, relevant validation.
- Do not claim success unless every acceptance criterion passes.

At completion, write a detailed report to:

reports/autonomous_loop/LATEST_CODEX_REPORT.md

The report must contain:

1. Objective
2. Files inspected
3. Files changed
4. Commands executed
5. Tests executed
6. Results
7. Failures
8. Evidence paths
9. Acceptance-criteria assessment
10. Remaining risks
11. Recommended next action

Controller iteration: {state["iteration"]}
""".strip()


def resolve_codex_command() -> str:
    candidates = ("codex.cmd", "codex.exe", "codex") if os.name == "nt" else (
        "codex",
    )
    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    raise RuntimeError("Codex CLI was not found on PATH.")


def scrub_codex_environment(
    source: Mapping[str, str] | None = None,
) -> dict[str, str]:
    environment = dict(source if source is not None else os.environ)
    # Repository-controlled commands must not inherit planner or unrelated
    # service credentials. Codex reuses its own saved login for this workflow.
    sensitive_exact = {
        "CODEX_API_KEY",
        "GITHUB_TOKEN",
        "GH_TOKEN",
        "OPENAI_API_KEY",
    }
    sensitive_fragments = (
        "ACCESS_KEY",
        "API_KEY",
        "AUTH_TOKEN",
        "CLIENT_SECRET",
        "CREDENTIAL",
        "PASSWORD",
        "PRIVATE_KEY",
        "SECRET",
    )
    for name in tuple(environment):
        normalized = name.upper()
        if normalized in sensitive_exact or any(
            fragment in normalized for fragment in sensitive_fragments
        ):
            environment.pop(name, None)
    return environment


def validate_codex_report(
    report_path: Path = LATEST_REPORT_FILE,
    previous_hash: str | None = None,
) -> tuple[bool, str]:
    if not report_path.is_file():
        return False, f"Codex did not create {report_path}."

    content = load_text(report_path)
    if not content.strip():
        return False, "Codex report is empty."

    current_hash = sha256_file(report_path)
    if previous_hash and current_hash == previous_hash:
        return False, "Codex report was not updated for this iteration."

    lowered = content.lower()
    missing = [
        section for section in REQUIRED_REPORT_SECTIONS if section not in lowered
    ]
    if missing:
        return False, f"Codex report is missing sections: {missing}"
    return True, current_hash


def run_codex(
    prompt: str,
    iteration: int,
    timeout_seconds: int,
) -> tuple[int, str, str, str]:
    event_file = RUNTIME_DIR / f"codex_events_{iteration:04d}.jsonl"
    last_message_file = (
        RUNTIME_DIR / f"codex_last_message_{iteration:04d}.txt"
    )
    previous_report_hash = (
        sha256_file(LATEST_REPORT_FILE)
        if LATEST_REPORT_FILE.is_file()
        else None
    )

    command = [
        resolve_codex_command(),
        "exec",
        "--json",
        "--ephemeral",
        "--sandbox",
        "workspace-write",
        "--output-last-message",
        str(last_message_file),
        "-",
    ]

    process = subprocess.run(
        command,
        cwd=ROOT,
        input=prompt,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
        env=scrub_codex_environment(),
    )

    write_text_atomic(event_file, process.stdout)
    last_message = load_text(last_message_file, process.stdout[-20_000:])

    report_ok, report_detail = validate_codex_report(
        previous_hash=previous_report_hash
    )
    return_code = process.returncode
    if return_code == 0 and not report_ok:
        return_code = 4

    return return_code, last_message, process.stderr, report_detail


def archive_iteration(
    iteration: int,
    decision: Mapping[str, Any],
    codex_last_message: str,
    stderr: str,
    report_validation: str,
) -> None:
    iteration_dir = REPORT_DIR / "raw" / f"iteration_{iteration:04d}"
    iteration_dir.mkdir(parents=True, exist_ok=True)

    write_text_atomic(
        iteration_dir / "planner_decision.json",
        json.dumps(dict(decision), indent=2, ensure_ascii=False) + "\n",
    )
    write_text_atomic(
        iteration_dir / "codex_last_message.txt",
        codex_last_message,
    )
    write_text_atomic(iteration_dir / "codex_stderr.txt", stderr)
    write_text_atomic(
        iteration_dir / "report_validation.txt",
        report_validation + "\n",
    )

    if LATEST_REPORT_FILE.exists():
        write_text_atomic(
            iteration_dir / "codex_report.md",
            load_text(LATEST_REPORT_FILE),
        )


def preflight() -> dict[str, Any]:
    verify_repository()
    codex_path = resolve_codex_command()
    codex_version = run_capture([codex_path, "--version"])
    return {
        "repository_root": str(ROOT),
        "git_repository": True,
        "working_tree_dirty": repository_is_dirty(),
        "master_goal_present": bool(load_text(MASTER_GOAL_FILE).strip()),
        "state_file_present": STATE_FILE.is_file(),
        "openai_api_key_configured": bool(os.getenv("OPENAI_API_KEY")),
        "codex_path": codex_path,
        "codex_version_exit_code": codex_version.returncode,
        "codex_version": codex_version.stdout.strip(),
        "protected_test_authorised": protected_test_approval_is_valid(
            load_state()
        ),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the local CHAAD planner-to-Codex controller."
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=DEFAULT_MAX_ITERATIONS,
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
    )
    parser.add_argument("--sleep", type=int, default=5)
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Explicitly allow Codex tasks in an already dirty worktree.",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Print local readiness checks without calling the API or Codex.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print a redacted planning-input summary without API/Codex calls.",
    )
    args = parser.parse_args(argv)

    if args.max_iterations < 1:
        parser.error("--max-iterations must be at least 1")
    if args.timeout < 1:
        parser.error("--timeout must be at least 1")
    if args.sleep < 0:
        parser.error("--sleep cannot be negative")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    ensure_directories()

    checks = preflight()
    if args.preflight:
        print(json.dumps(checks, indent=2))
        return 0

    master_goal = load_text(MASTER_GOAL_FILE)
    if not master_goal.strip():
        print(f"Missing or empty master goal: {MASTER_GOAL_FILE}", file=sys.stderr)
        return 2

    state = load_state()
    if args.dry_run:
        print(
            json.dumps(
                {
                    "mode": "DRY_RUN",
                    "planner_model": PLANNER_MODEL,
                    "preflight": checks,
                    "state": state,
                    "master_goal_chars": len(master_goal),
                    "latest_report_present": LATEST_REPORT_FILE.is_file(),
                    "would_call_api": False,
                    "would_call_codex": False,
                },
                indent=2,
            )
        )
        return 0

    if checks["working_tree_dirty"] and not args.allow_dirty:
        print(
            "Working tree is dirty. Review and preserve existing changes, then "
            "rerun with --allow-dirty only if autonomous edits are acceptable.",
            file=sys.stderr,
        )
        return 2

    if not checks["openai_api_key_configured"]:
        print("OPENAI_API_KEY is not configured.", file=sys.stderr)
        return 2

    try:
        from openai import OpenAI
    except ImportError:
        print(
            "The openai package is not installed. Install dependencies from "
            "orchestrator/requirements.txt in an isolated environment.",
            file=sys.stderr,
        )
        return 2

    client = OpenAI(timeout=PLANNER_TIMEOUT_SECONDS, max_retries=2)

    for _ in range(args.max_iterations):
        state["iteration"] += 1
        state["status"] = "PLANNING"
        state["human_approval_required"] = False
        state["stop_reason"] = ""
        state["updated_at"] = utc_now()
        save_json_atomic(STATE_FILE, state)

        current_report = load_text(
            LATEST_REPORT_FILE,
            "No Codex task has been completed yet.",
        )

        try:
            decision = ask_planner(
                client=client,
                master_goal=master_goal,
                state=state,
                codex_report=current_report,
                repository_snapshot=git_snapshot(),
            )
        except Exception as exc:
            state["status"] = "BLOCKED"
            state["stop_reason"] = (
                f"Planner request or response validation failed: {exc}"
            )
            state["updated_at"] = utc_now()
            save_json_atomic(STATE_FILE, state)
            print(state["stop_reason"], file=sys.stderr)
            return 3

        decision = apply_human_gates(decision, state)
        state["next_action"] = decision["title"]
        state["current_phase"] = decision["phase"]
        state["updated_at"] = utc_now()

        if decision["action_type"] != "CODEX_TASK":
            state["status"] = decision["action_type"]
            state["human_approval_required"] = (
                decision["action_type"] == "HUMAN_APPROVAL"
            )
            state["stop_reason"] = (
                decision["human_decision_required"] or decision["reason"]
            )
            state["history"].append(
                {
                    "iteration": state["iteration"],
                    "decision": decision,
                    "timestamp": utc_now(),
                }
            )
            save_json_atomic(STATE_FILE, state)
            print(json.dumps(decision, indent=2))
            return 0

        state["status"] = "CODEX_RUNNING"
        save_json_atomic(STATE_FILE, state)

        try:
            return_code, last_message, stderr, report_validation = run_codex(
                prompt=codex_prompt(decision, state),
                iteration=state["iteration"],
                timeout_seconds=args.timeout,
            )
        except subprocess.TimeoutExpired:
            state["status"] = "BLOCKED"
            state["stop_reason"] = (
                f"Codex timed out after {args.timeout} seconds."
            )
            state["updated_at"] = utc_now()
            save_json_atomic(STATE_FILE, state)
            print(state["stop_reason"], file=sys.stderr)
            return 3

        archive_iteration(
            iteration=state["iteration"],
            decision=decision,
            codex_last_message=last_message,
            stderr=stderr,
            report_validation=report_validation,
        )

        state["last_action"] = decision["title"]
        state["status"] = (
            "CODEX_REPORTED" if return_code == 0 else "CODEX_FAILED"
        )
        state["history"].append(
            {
                "iteration": state["iteration"],
                "phase": decision["phase"],
                "task": decision["title"],
                "codex_return_code": return_code,
                "report": str(LATEST_REPORT_FILE.relative_to(ROOT)),
                "report_validation": report_validation,
                "timestamp": utc_now(),
            }
        )
        state["updated_at"] = utc_now()
        save_json_atomic(STATE_FILE, state)

        if return_code != 0:
            print(
                f"Codex failed with exit code {return_code}. "
                "Review the archived iteration before continuing.",
                file=sys.stderr,
            )
            return 4

        time.sleep(args.sleep)

    state["status"] = "ITERATION_LIMIT_REACHED"
    state["stop_reason"] = (
        f"Stopped after {args.max_iterations} iterations."
    )
    state["updated_at"] = utc_now()
    save_json_atomic(STATE_FILE, state)
    print(state["stop_reason"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
