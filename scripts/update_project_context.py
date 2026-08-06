#!/usr/bin/env python3
"""
scripts/update_project_context.py
────────────────────────────────────────────────────────
Safe context maintenance script for CHAAD project.

Collects repository metadata and updates auto-generated
sections in project documentation. Never modifies manually
written content — only text between markers:

    <!-- AUTO-GENERATED-CONTEXT:START -->
    ...
    <!-- AUTO-GENERATED-CONTEXT:END -->

Usage:
    python scripts/update_project_context.py --check     # Dry run
    python scripts/update_project_context.py --update    # Apply updates
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ── Constants ───────────────────────────────────────────

START_MARKER = "<!-- AUTO-GENERATED-CONTEXT:START -->"
END_MARKER = "<!-- AUTO-GENERATED-CONTEXT:END -->"

# Files with auto-generated sections
MANAGED_FILES = [
    "docs/CURRENT_STATE.md",
    "chatgpt_context/CHATGPT_CURRENT_HANDOFF.md",
]


# ── Git Helpers ─────────────────────────────────────────

def _run_git(args: List[str], cwd: Optional[str] = None) -> Tuple[int, str, str]:
    """Run a git command safely. Returns (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(
            ["git"] + args,
            capture_output=True,
            text=True,
            cwd=cwd or os.getcwd(),
            timeout=10,
        )
        return result.returncode, result.stdout.strip(), result.stderr.strip()
    except FileNotFoundError:
        return -1, "", "git command not found"
    except subprocess.TimeoutExpired:
        return -1, "", "git command timed out"
    except Exception as e:
        return -1, "", str(e)


def get_git_branch(project_root: str) -> Optional[str]:
    """Get current Git branch name."""
    rc, out, _ = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=project_root)
    return out if rc == 0 and out else None


def get_git_commit(project_root: str) -> Optional[str]:
    """Get current Git commit hash."""
    rc, out, _ = _run_git(["rev-parse", "--short", "HEAD"], cwd=project_root)
    return out if rc == 0 and out else None


def get_git_status_summary(project_root: str) -> Dict[str, int]:
    """Count staged, unstaged, and untracked files."""
    summary = {"staged": 0, "unstaged": 0, "untracked": 0}
    rc, out, _ = _run_git(["status", "--porcelain"], cwd=project_root)
    if rc != 0:
        return summary
    for line in out.split("\n"):
        line = line.strip()
        if not line:
            continue
        # First two chars are status: XY where X=staging, Y=working
        if len(line) >= 2:
            x = line[0]
            y = line[1]
            if x != " " and x != "?":
                summary["staged"] += 1
            if y != " " and y != "?":
                summary["unstaged"] += 1
            if x == "?" and y == "?":
                summary["untracked"] += 1
    return summary


def is_git_repo(project_root: str) -> bool:
    """Check if the directory is a Git repository."""
    rc, _, _ = _run_git(["rev-parse", "--git-dir"], cwd=project_root)
    return rc == 0


# ── Documentation File Helpers ──────────────────────────

def get_python_version() -> str:
    """Get the Python version string."""
    return sys.version.split()[0]


def count_files_in_dir(directory: str, pattern: str = "*.py") -> int:
    """Count files matching a pattern in a directory tree."""
    try:
        return len(list(Path(directory).rglob(pattern)))
    except Exception:
        return -1


def check_expected_files(project_root: str) -> Tuple[List[str], List[str]]:
    """Check which expected documentation files exist and which are missing."""
    expected = [
        "AGENTS.md",
        "docs/AI_CONTEXT_INDEX.md",
        "docs/PROJECT_CONTEXT.md",
        "docs/CURRENT_STATE.md",
        "docs/ROADMAP.md",
        "docs/PROJECT_HISTORY.md",
        "docs/KNOWN_ISSUES.md",
        "docs/DECISIONS.md",
        "docs/ENVIRONMENT.md",
        "docs/TESTING_AND_VALIDATION.md",
        "docs/DATASET_AND_SPLITS.md",
        "docs/EXPERIMENT_LOG.md",
        "docs/MODEL_CARD.md",
        "docs/METRICS_REGISTRY.md",
        "docs/DATA_LEAKAGE_AUDIT.md",
        "docs/REPRODUCIBILITY.md",
        "docs/SESSION_LOG.md",
        "chatgpt_context/CHATGPT_MASTER_CONTEXT.md",
        "chatgpt_context/CHATGPT_CURRENT_HANDOFF.md",
        "chatgpt_context/CHATGPT_START_PROMPT.md",
        "chatgpt_context/CHATGPT_END_SESSION_PROMPT.md",
        "chatgpt_context/CHATGPT_TASK_TEMPLATE.md",
    ]
    existing = []
    missing = []
    for f in expected:
        full = os.path.join(project_root, f)
        if os.path.isfile(full) and os.path.getsize(full) > 0:
            existing.append(f)
        else:
            missing.append(f)
    return existing, missing


# ── Auto-Generated Content Generator ────────────────────

def generate_metadata_content(project_root: str) -> str:
    """Generate the auto-updated metadata section content."""
    lines = []
    lines.append(f"| Field | Value |")
    lines.append(f"|-------|-------|")
    lines.append(f"| Date | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} |")

    if is_git_repo(project_root):
        branch = get_git_branch(project_root) or "UNKNOWN"
        commit = get_git_commit(project_root) or "UNKNOWN"
        status = get_git_status_summary(project_root)
        lines.append(f"| Branch | `{branch}` |")
        lines.append(f"| Commit | `{commit}` |")
        lines.append(f"| Staged | {status['staged']} |")
        lines.append(f"| Unstaged | {status['unstaged']} |")
        lines.append(f"| Untracked | {status['untracked']} |")
    else:
        lines.append(f"| Git | Not a Git repository |")

    lines.append(f"| Python | {get_python_version()} |")
    return "\n".join(lines)


def update_file(filepath: str, new_content: str) -> bool:
    """
    Update auto-generated section in a file.

    Only modifies text between START_MARKER and END_MARKER.
    Returns True if the file was modified.
    """
    full_path = Path(filepath)
    if not full_path.exists():
        print(f"  [SKIP] File not found: {filepath}")
        return False

    content = full_path.read_text(encoding="utf-8")

    if START_MARKER not in content or END_MARKER not in content:
        print(f"  [SKIP] No auto-generated markers in: {filepath}")
        return False

    # Replace content between markers
    before = content[: content.find(START_MARKER) + len(START_MARKER)]
    after = content[content.find(END_MARKER) :]

    new_full = before + "\n" + new_content + "\n" + after

    if new_full == content:
        print(f"  [OK] No changes needed: {filepath}")
        return False

    full_path.write_text(new_full, encoding="utf-8")
    print(f"  [UPDATED] {filepath}")
    return True


# ── Main ────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Update auto-generated context in CHAAD project documentation."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check what would change without modifying files.",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Apply updates to managed files.",
    )
    parser.add_argument(
        "--root",
        default=None,
        help="Project root directory (default: parent of this script).",
    )
    args = parser.parse_args()

    if not args.check and not args.update:
        print("Usage: python scripts/update_project_context.py --check | --update")
        return 1

    # Determine project root
    if args.root:
        project_root = os.path.abspath(args.root)
    else:
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..")
        )

    os.chdir(project_root)

    print(f"Project root: {project_root}")
    print(f"Mode: {'DRY RUN' if args.check else 'UPDATE'}")

    # Check expected files
    existing, missing = check_expected_files(project_root)
    if missing:
        print(f"\n⚠ Missing documentation files ({len(missing)}):")
        for f in missing:
            print(f"  - {f}")

    # Generate metadata content
    meta_content = generate_metadata_content(project_root)
    print(f"\nGenerated metadata:")
    print(meta_content)

    # Check managed files for markers
    all_have_markers = True
    for mf in MANAGED_FILES:
        fp = os.path.join(project_root, mf)
        if not os.path.isfile(fp):
            print(f"\n  [WARNING] Managed file missing: {mf}")
            all_have_markers = False
        else:
            content = Path(fp).read_text(encoding="utf-8")
            if START_MARKER not in content:
                print(f"\n  [WARNING] Missing START marker in: {mf}")
                all_have_markers = False
            if END_MARKER not in content:
                print(f"\n  [WARNING] Missing END marker in: {mf}")
                all_have_markers = False

    # Update files
    if args.update:
        print(f"\nUpdating managed files...")
        any_changed = False
        for mf in MANAGED_FILES:
            fp = os.path.join(project_root, mf)
            if update_file(fp, meta_content):
                any_changed = True

        if not any_changed:
            print(f"  No files needed updating.")

    # Final status
    exit_code = 0
    if missing:
        exit_code = 2  # Missing files
        print(f"\n⚠ {len(missing)} expected documentation files are missing.")
    if not all_have_markers:
        exit_code = max(exit_code, 1)

    if exit_code == 0:
        print(f"\n✅ All checks passed.")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
