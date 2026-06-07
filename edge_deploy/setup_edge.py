"""
edge_deploy/setup_edge.py
────────────────────────────────────────────────────────
One-shot setup script to:
  1. Install dev-machine dependencies (onnxruntime, onnx, onnxsim)
  2. Run the full export + quantization pipeline
  3. Verify the exported models load correctly
  4. Print the complete deployment command sequence for RPi5

Usage (dev machine — must have PyTorch + project env active):
    python edge_deploy/setup_edge.py
    python edge_deploy/setup_edge.py --checkpoint checkpoints/best_model.pt
    python edge_deploy/setup_edge.py --skip-int8   # FP32/FP16 only, no ORT quantization needed
"""

from __future__ import annotations

import os
import sys
import subprocess
import argparse
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
EDGE_DIR = str(Path(__file__).resolve().parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ─────────────────────────────────────────────────────────
#  Dependency Installation
# ─────────────────────────────────────────────────────────

DEV_DEPS = [
    "onnx>=1.14.0",
    "onnxruntime>=1.17.0",
    "onnxsim>=0.4.33",
    "onnxconverter-common>=1.13.0",
]


def install_deps(deps: list):
    """Install Python packages using pip."""
    for dep in deps:
        print(f"[Setup] Installing {dep}...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", dep, "-q"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"  WARNING: {dep} install failed: {result.stderr.strip()}")
        else:
            print(f"  ✓ {dep}")


# ─────────────────────────────────────────────────────────
#  Model Verification
# ─────────────────────────────────────────────────────────

def verify_onnx_models(output_dir: str) -> dict:
    """Load and verify each exported ONNX model."""
    results = {}
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        print("[Verify] onnx/onnxruntime not available")
        return {}

    for fname in os.listdir(output_dir):
        if not fname.endswith(".onnx"):
            continue
        fpath = os.path.join(output_dir, fname)
        try:
            # Structure check
            model = onnx.load(fpath)
            onnx.checker.check_model(model)

            # Runtime load check
            sess = ort.InferenceSession(fpath, providers=["CPUExecutionProvider"])
            inp = sess.get_inputs()[0]
            out_names = [o.name for o in sess.get_outputs()]

            # Quick inference check — resolve dynamic dims
            shape = [s if isinstance(s, int) else 1 for s in inp.shape]
            import numpy as np
            x = np.random.randn(*shape).astype(np.float32)
            x = np.clip(x, 0, 1)
            outputs = sess.run(None, {inp.name: x})

            size_mb = os.path.getsize(fpath) / (1024 * 1024)
            results[fname] = {
                "status": "OK",
                "size_mb": round(size_mb, 1),
                "input": f"{inp.name} {inp.shape}",
                "outputs": out_names,
                "output_shapes": [str(o.shape) for o in outputs],
            }
            print(f"  ✓ {fname} ({size_mb:.1f} MB) — inference OK")

        except Exception as e:
            results[fname] = {"status": f"FAILED: {e}"}
            print(f"  ✗ {fname} — {e}")

    return results


# ─────────────────────────────────────────────────────────
#  Deployment Instructions
# ─────────────────────────────────────────────────────────

def print_deployment_guide(output_dir: str, rpi_host: str = "pi@raspberrypi.local"):
    """Print the complete RPi5 deployment command sequence."""
    abs_output = os.path.abspath(output_dir)
    print("\n" + "=" * 60)
    print("  RASPBERRY PI 5 DEPLOYMENT GUIDE")
    print("=" * 60)

    print("""
━━━ STEP 1: Copy files to RPi5 ━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
    print(f"  scp -r {abs_output}/* {rpi_host}:/opt/mimii/models/")
    for fname in (
        "edge_streaming.py",
        "edge_runtime.py",
        "edge_watchdog.py",
        "edge_benchmark.py",
        "inference_server.py",
        "auto_update.py",
        "config.yaml",
        "requirements_rpi.txt",
        "requirements_edge.txt",
        "deploy.sh",
    ):
        print(f"  scp {EDGE_DIR}/{fname} {rpi_host}:/opt/mimii/")
    print(f"  scp -r {EDGE_DIR}/systemd {rpi_host}:/opt/mimii/")

    print("""
━━━ STEP 2: RPi5 system setup (run on RPi5) ━━━━━━━━━━━━━━
""")
    print("""  # Create user + dirs
  sudo useradd -m -G audio mimii
  sudo mkdir -p /opt/mimii /var/log/mimii
  sudo chown -R mimii:mimii /opt/mimii /var/log/mimii

  # Python venv
  cd /opt/mimii
  python3 -m venv venv
  source venv/bin/activate
  pip install -r requirements_rpi.txt
  chmod +x deploy.sh

  # CPU governor: lock to performance
  echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

  # GPU memory: minimize (no display needed)
  echo "gpu_mem=16" | sudo tee -a /boot/firmware/config.txt

  # Disable unnecessary services
  sudo systemctl disable bluetooth avahi-daemon""")

    print("""
━━━ STEP 3: Benchmark (run on RPi5) ━━━━━━━━━━━━━━━━━━━━━━
""")
    print("""  cd /opt/mimii
  source venv/bin/activate

  # Latency benchmark (INT8 model)
  python3 edge_benchmark.py \\
      --model models/classifier_int8.onnx \\
      --warmup 20 --iterations 200

  # Mel transform benchmark
  python3 edge_benchmark.py --mel

  # FP32 vs INT8 accuracy validation
  python3 edge_benchmark.py --validate \\
      --fp32 models/classifier_fp32.onnx \\
      --int8 models/classifier_int8.onnx""")

    print("""
━━━ STEP 4: Start streaming (run on RPi5) ━━━━━━━━━━━━━━━━
""")
    print("""  # List audio devices first
  python3 edge_streaming.py --list-devices

  # Start streaming (test mode, direct)
  python3 edge_streaming.py --config config.yaml

  # Install as systemd services
  sudo cp /opt/mimii/systemd/mimii-detector.service /etc/systemd/system/
  sudo cp /opt/mimii/systemd/mimii-api.service /etc/systemd/system/
  sudo systemctl daemon-reload
  sudo systemctl enable --now mimii-detector   # 24/7 mic streaming
  sudo systemctl enable --now mimii-api        # optional HTTP API :8000

  # Monitor
  journalctl -fu mimii-detector
  curl http://localhost:8000/health
  python3 edge_watchdog.py --once

  # OTA model swap (optional background watcher)
  nohup python3 auto_update.py &""")

    print("""
━━━ STEP 5: Thermal soak test (optional) ━━━━━━━━━━━━━━━━━
""")
    print("""  python3 edge_benchmark.py \\
      --model models/classifier_int8.onnx \\
      --soak-hours 4 \\
      --soak-log-interval 30
  # Monitor: soak_results.csv""")

    print("\n" + "=" * 60)


# ─────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Edge deployment setup — export, quantize, verify"
    )
    parser.add_argument(
        "--checkpoint", default="checkpoints/best_model.pt",
        help="PyTorch checkpoint path"
    )
    parser.add_argument(
        "--output", default="edge_deploy/models",
        help="Output directory for exported models"
    )
    parser.add_argument(
        "--calib-data", default=None,
        help="Directory of real .npy mel calibration samples"
    )
    parser.add_argument(
        "--n-calib", type=int, default=200,
        help="Number of calibration samples"
    )
    parser.add_argument(
        "--skip-deps", action="store_true",
        help="Skip pip install of dev dependencies"
    )
    parser.add_argument(
        "--skip-int8", action="store_true",
        help="Skip INT8 quantization"
    )
    parser.add_argument(
        "--skip-validation", action="store_true",
        help="Skip FP32 vs INT8 validation"
    )
    parser.add_argument(
        "--rpi-host", default="pi@raspberrypi.local",
        help="RPi5 SSH host for deployment guide"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  MIMII Edge Deployment Setup")
    print("=" * 60)
    print(f"  Checkpoint:  {args.checkpoint}")
    print(f"  Output dir:  {args.output}")
    print(f"  INT8 PTQ:    {'SKIP' if args.skip_int8 else 'YES'}")
    print("=" * 60 + "\n")

    # Step 1: Install dependencies
    if not args.skip_deps:
        print("[Setup] Installing development dependencies...")
        install_deps(DEV_DEPS)
        print()

    # Step 2: Run export pipeline
    print("[Setup] Running export + quantization pipeline...")
    from edge_quantize import run_pipeline
    run_pipeline(
        checkpoint_path=args.checkpoint,
        output_dir=args.output,
        calib_data_dir=args.calib_data,
        n_calib_samples=args.n_calib,
        skip_int8=args.skip_int8,
        skip_validation=args.skip_validation,
    )

    # Step 3: Verify models
    print("\n[Setup] Verifying exported models...")
    verify_results = verify_onnx_models(args.output)

    ok_count = sum(1 for v in verify_results.values() if v.get("status") == "OK")
    total = len(verify_results)
    print(f"\n[Setup] Verification: {ok_count}/{total} models OK")

    if ok_count < total:
        print("[Setup] WARNING: Some models failed verification — check output above")

    # Step 4: Print deployment guide
    print_deployment_guide(args.output, args.rpi_host)


if __name__ == "__main__":
    main()
