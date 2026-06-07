"""
deploy.py
────────────────────────────────────────────────────────
Deployment automation script.

Handles:
  - Environment setup
  - Model export to production formats
  - Docker build and push
  - Health checks
  - Rollback capabilities
"""

from __future__ import annotations

import os
import sys
import subprocess
import argparse
import json
from pathlib import Path
from typing import Optional, List

import torch

sys.path.insert(0, str(Path(__file__).parent))

from config import cfg
from models.hybrid_model import HybridAnomalyModel
from utils.export import export_all_formats
from utils.gpu_utils import setup_cuda_optimizations
from utils.validation import setup_logging, logger, system_health_check


def run_command(cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command with logging."""
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.stdout:
        logger.info(result.stdout)
    if result.stderr:
        logger.warning(result.stderr)

    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")

    return result


def check_prerequisites():
    """Check that all prerequisites are met."""
    logger.info("Checking prerequisites...")

    # Check CUDA
    if not torch.cuda.is_available():
        logger.warning("CUDA not available. Some optimizations will be skipped.")

    # Check Docker
    try:
        run_command(["docker", "--version"], check=False)
        logger.info("Docker is available")
    except FileNotFoundError:
        logger.error("Docker not found. Install Docker for containerized deployment.")

    # Check git
    try:
        run_command(["git", "--version"], check=False)
        logger.info("Git is available")
    except FileNotFoundError:
        logger.warning("Git not found")

    # System health
    health = system_health_check()
    if health["status"] != "healthy":
        logger.warning(f"System health issues: {health.get('issues', [])}")


def export_model():
    """Export model to production formats."""
    logger.info("Exporting model to production formats...")

    setup_cuda_optimizations()

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridAnomalyModel(cfg.model).to(device)

    ckpt_path = os.path.join(cfg.training.checkpoint_dir, "best_model.pt")
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        logger.info(f"Loaded checkpoint from {ckpt_path}")
    else:
        logger.warning(f"No checkpoint found at {ckpt_path}")
        return

    # Export
    results = export_all_formats(model, cfg, output_dir="exports")

    logger.info("Export results:")
    for fmt, path in results.items():
        size_mb = os.path.getsize(path) / (1024 * 1024)
        logger.info(f"  {fmt}: {path} ({size_mb:.2f} MB)")


def build_docker(image_tag: str = "mimii-anomaly-detector:latest"):
    """Build Docker image."""
    logger.info(f"Building Docker image: {image_tag}")

    run_command([
        "docker", "build",
        "-t", image_tag,
        "-f", "Dockerfile",
        "--target", "production",
        "."
    ])

    logger.info(f"Successfully built {image_tag}")


def push_docker(image_tag: str, registry: Optional[str] = None):
    """Push Docker image to registry."""
    if registry:
        full_tag = f"{registry}/{image_tag}"
        run_command(["docker", "tag", image_tag, full_tag])
    else:
        full_tag = image_tag

    logger.info(f"Pushing {full_tag}...")
    run_command(["docker", "push", full_tag])


def run_docker(
    image_tag: str = "mimii-anomaly-detector:latest",
    port: int = 7860,
    gpu: bool = True,
):
    """Run Docker container locally."""
    logger.info(f"Running Docker container: {image_tag}")

    cmd = [
        "docker", "run", "-d",
        "--name", "mimii-detector",
        "-p", f"{port}:{port}",
    ]

    if gpu:
        cmd.extend(["--gpus", "all"])

    cmd.extend([
        "-v", f"{os.path.abspath('checkpoints')}:/app/checkpoints:ro",
        "-v", f"{os.path.abspath('logs')}:/app/logs",
        image_tag
    ])

    run_command(cmd)
    logger.info(f"Container running on port {port}")


def stop_docker():
    """Stop and remove Docker container."""
    logger.info("Stopping Docker container...")
    run_command(["docker", "stop", "mimii-detector"], check=False)
    run_command(["docker", "rm", "mimii-detector"], check=False)


def health_check_container():
    """Health check the running container."""
    logger.info("Running health check...")

    result = run_command([
        "docker", "exec", "mimii-detector",
        "python", "-c",
        "import torch; print('GPU:', torch.cuda.is_available())"
    ], check=False)

    if "True" in result.stdout:
        logger.info("Health check passed: GPU available")
        return True
    else:
        logger.error("Health check failed: GPU not available")
        return False


def create_deployment_bundle(version: str):
    """Create a deployment bundle."""
    logger.info(f"Creating deployment bundle version {version}...")

    bundle_dir = f"deployments/mimii-detector-{version}"
    os.makedirs(bundle_dir, exist_ok=True)

    # Copy essential files
    files_to_copy = [
        "app.py", "config.py", "requirements.txt", "Dockerfile",
        "docker-compose.yml", "production_api.py", "benchmark.py",
        "models/", "inference/", "utils/", "data/", "training/",
    ]

    for item in files_to_copy:
        dest = os.path.join(bundle_dir, item)
        if os.path.isdir(item):
            run_command(["cp", "-r", item, dest], check=False)
        else:
            run_command(["cp", item, dest], check=False)

    # Create metadata
    metadata = {
        "version": version,
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "files": files_to_copy,
    }

    with open(os.path.join(bundle_dir, "deployment.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Deployment bundle created at {bundle_dir}")


def main():
    parser = argparse.ArgumentParser(description="Deploy MIMII Anomaly Detection")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Check command
    subparsers.add_parser("check", help="Check prerequisites")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export model")

    # Docker build command
    build_parser = subparsers.add_parser("build", help="Build Docker image")
    build_parser.add_argument("--tag", default="mimii-anomaly-detector:latest")

    # Docker push command
    push_parser = subparsers.add_parser("push", help="Push Docker image")
    push_parser.add_argument("--tag", default="mimii-anomaly-detector:latest")
    push_parser.add_argument("--registry", help="Docker registry URL")

    # Docker run command
    run_parser = subparsers.add_parser("run", help="Run Docker container")
    run_parser.add_argument("--tag", default="mimii-anomaly-detector:latest")
    run_parser.add_argument("--port", type=int, default=7860)
    run_parser.add_argument("--no-gpu", action="store_true")

    # Stop command
    subparsers.add_parser("stop", help="Stop Docker container")

    # Health check command
    subparsers.add_parser("health", help="Health check container")

    # Bundle command
    bundle_parser = subparsers.add_parser("bundle", help="Create deployment bundle")
    bundle_parser.add_argument("--version", required=True)

    # Full deployment command
    full_parser = subparsers.add_parser("deploy", help="Full deployment")
    full_parser.add_argument("--version", required=True)
    full_parser.add_argument("--tag", default="mimii-anomaly-detector:latest")

    args = parser.parse_args()

    setup_logging()

    if args.command == "check":
        check_prerequisites()

    elif args.command == "export":
        export_model()

    elif args.command == "build":
        check_prerequisites()
        export_model()
        build_docker(args.tag)

    elif args.command == "push":
        push_docker(args.tag, args.registry)

    elif args.command == "run":
        run_docker(args.tag, args.port, not args.no_gpu)

    elif args.command == "stop":
        stop_docker()

    elif args.command == "health":
        health_check_container()

    elif args.command == "bundle":
        create_deployment_bundle(args.version)

    elif args.command == "deploy":
        # Full deployment pipeline
        check_prerequisites()
        export_model()
        build_docker(args.tag)
        stop_docker()  # Stop existing if running
        run_docker(args.tag)

        if health_check_container():
            logger.info("Deployment successful!")
            create_deployment_bundle(args.version)
        else:
            logger.error("Deployment failed health check")
            stop_docker()
            sys.exit(1)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
