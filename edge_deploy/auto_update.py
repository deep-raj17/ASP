"""
edge_deploy/auto_update.py
────────────────────────────────────────────────────────
Watch classifier ONNX for changes and restart inference services.

Run as a background process or lightweight systemd unit on RPi5:

    python3 auto_update.py
    python3 auto_update.py --model models/classifier_int8.onnx --interval 10
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path


def file_hash(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def restart_services(services: list[str]) -> None:
    for svc in services:
        try:
            subprocess.run(
                ["sudo", "systemctl", "restart", svc],
                check=True,
                capture_output=True,
                text=True,
            )
            print(f"[auto_update] Restarted {svc}")
        except subprocess.CalledProcessError as e:
            print(f"[auto_update] Failed to restart {svc}: {e.stderr.strip()}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Watch ONNX model and restart services")
    parser.add_argument(
        "--model",
        default=os.environ.get(
            "MIMII_CLASSIFIER",
            "models/classifier_int8.onnx",
        ),
        help="Path to classifier ONNX",
    )
    parser.add_argument("--interval", type=int, default=10, help="Poll interval (seconds)")
    parser.add_argument(
        "--services",
        nargs="+",
        default=["mimii-detector", "mimii-api"],
        help="systemd units to restart on model change",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.is_file():
        print(f"[auto_update] Model not found: {model_path.resolve()}", file=sys.stderr)
        sys.exit(1)

    print(f"[auto_update] Watching {model_path.resolve()} every {args.interval}s")
    old_hash = file_hash(model_path)

    while True:
        time.sleep(args.interval)
        try:
            new_hash = file_hash(model_path)
        except OSError:
            continue
        if new_hash != old_hash:
            print("[auto_update] New model detected — restarting services")
            restart_services(args.services)
            old_hash = new_hash


if __name__ == "__main__":
    main()
