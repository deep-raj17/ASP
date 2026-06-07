"""Safe cleanup helper for the inference project.

Keeps trained, calibrated, evaluated, and exported model artifacts. Removes only
generated caches, broken duplicate virtual environments, temporary logs, and
stray junk files.
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROTECTED_DIRS = {
    ROOT / "checkpoints",
    ROOT / "artifacts",
    ROOT / "edge_deploy" / "models",
}


def _inside_project(path: Path) -> bool:
    try:
        path.resolve().relative_to(ROOT)
        return True
    except ValueError:
        return False


def _remove_path(path: Path) -> bool:
    if not _inside_project(path):
        print(f"skip outside project: {path}")
        return False
    for protected in PROTECTED_DIRS:
        if path.resolve() == protected.resolve() or protected.resolve() in path.resolve().parents:
            print(f"skip protected artifact path: {path}")
            return False
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
        return True
    if path.exists():
        try:
            path.unlink(missing_ok=True)
            return True
        except OSError as exc:
            print(f"skip {path}: {exc}")
            return False
    return False


def _venv_broken() -> bool:
    cfg = ROOT / "venv" / "pyvenv.cfg"
    py = ROOT / "venv" / "Scripts" / "python.exe"
    if not cfg.is_file() or not py.is_file():
        return True
    text = cfg.read_text(encoding="utf-8", errors="ignore")
    username = os.environ.get("USERNAME", "").lower()
    return "deepr" in text.lower() and username != "deepr"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Remove generated project files safely.")
    parser.add_argument(
        "--remove-venv",
        action="store_true",
        help="Also remove the local venv folder. Recreate it later with setup.bat.",
    )
    args = parser.parse_args(argv)

    print("MIMII cleanup: preserving trained inference artifacts.")
    removed = 0

    for pattern in [
        "checkpoints/epoch_*.pt",
        "logs/*.log",
        "logs/events.out.tfevents.*",
    ]:
        for item in glob.glob(str(ROOT / pattern)):
            removed += int(_remove_path(Path(item)))

    for name in [".venv", ".venv310", "venv311", "venv.broken-20260514-131750", ".sixth"]:
        removed += int(_remove_path(ROOT / name))

    if args.remove_venv or _venv_broken():
        removed += int(_remove_path(ROOT / "venv"))

    for cache_dir in ROOT.rglob("__pycache__"):
        if "venv" not in cache_dir.parts:
            removed += int(_remove_path(cache_dir))

    removed += int(_remove_path(ROOT / "nul"))
    print(f"Cleanup complete. Removed {removed} generated item(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
