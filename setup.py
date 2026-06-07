#!/usr/bin/env python3
"""
setup.py
────────────────────────────────────────────────────────
Cross-platform environment setup script.

Creates virtual environment and installs all dependencies.
Handles PyTorch CUDA installation automatically.

Usage:
    python setup.py              # Full setup
    python setup.py --venv venv  # Custom venv name
    python setup.py --cpu        # CPU-only mode (no CUDA)
"""

import os
import sys
import subprocess
import argparse
import platform
from pathlib import Path


def run_command(cmd, check=True, capture=False):
    """Run a shell command."""
    print(f"Running: {' '.join(cmd)}")
    
    if capture:
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result
    
    result = subprocess.run(cmd, check=check)
    return result


def get_python_executable():
    """Get the current Python executable."""
    return sys.executable


def create_virtual_environment(venv_name="venv"):
    """Create virtual environment."""
    print(f"\n{'='*60}")
    print(f"Creating virtual environment: {venv_name}")
    print(f"{'='*60}\n")
    
    venv_path = Path(venv_name)
    
    if venv_path.exists():
        print(f"⚠️  Virtual environment '{venv_name}' already exists.")
        response = input("Delete and recreate? (y/n): ").lower().strip()
        if response == 'y':
            import shutil
            shutil.rmtree(venv_path)
        else:
            print("Using existing environment.")
            return venv_path
    
    # Create venv
    run_command([get_python_executable(), "-m", "venv", venv_name])
    
    print(f"✅ Virtual environment created at: {venv_path.absolute()}")
    return venv_path


def get_venv_python(venv_path):
    """Get Python executable path in venv."""
    if platform.system() == "Windows":
        return venv_path / "Scripts" / "python.exe"
    else:
        return venv_path / "bin" / "python"


def get_venv_pip(venv_path):
    """Get pip executable path in venv."""
    if platform.system() == "Windows":
        return venv_path / "Scripts" / "pip.exe"
    else:
        return venv_path / "bin" / "pip"


def upgrade_pip(venv_path):
    """Upgrade pip in virtual environment."""
    print("\n📦 Upgrading pip...")
    python = get_venv_python(venv_path)
    # Use python -m pip to avoid Windows pip.exe locking issues
    run_command([str(python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])


def install_pytorch(venv_path, cpu_only=False):
    """Install PyTorch with appropriate CUDA version."""
    print(f"\n{'='*60}")
    if cpu_only:
        print("Installing PyTorch (CPU version)")
    else:
        print("Installing PyTorch with CUDA 12.1 support")
    print(f"{'='*60}\n")
    
    python = get_venv_python(venv_path)
    
    if cpu_only:
        # CPU-only installation
        cmd = [
            str(python), "-m", "pip", "install",
            "torch", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cpu"
        ]
    else:
        # CUDA 12.1 installation
        cmd = [
            str(python), "-m", "pip", "install",
            "torch", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cu121"
        ]
    
    run_command(cmd)


def install_requirements(venv_path, req_file="requirements.txt"):
    """Install requirements from file."""
    print(f"\n{'='*60}")
    print(f"Installing requirements from {req_file}")
    print(f"{'='*60}\n")
    
    python = get_venv_python(venv_path)
    
    if not Path(req_file).exists():
        print(f"⚠️  {req_file} not found. Skipping.")
        return
    
    run_command([str(python), "-m", "pip", "install", "-r", req_file])


def install_optional_dependencies(venv_path):
    """Install optional dependencies for production."""
    print(f"\n{'='*60}")
    print("Installing optional dependencies")
    print(f"{'='*60}\n")
    
    python = get_venv_python(venv_path)
    
    optional_packages = [
        # Export & optimization (optional)
        # Uncomment if needed:
        # "onnx",
        # "onnxruntime-gpu",
        # "onnxsim",
        # "tensorrt",
        # "pycuda",
        
        # Development & testing
        "pytest",
        "black",
        "flake8",
    ]
    
    for package in optional_packages:
        print(f"\nInstalling {package}...")
        try:
            run_command([str(python), "-m", "pip", "install", package], check=False)
        except Exception as e:
            print(f"⚠️  Failed to install {package}: {e}")


def verify_installation(venv_path, cpu_only=False):
    """Verify that all dependencies are installed correctly."""
    print(f"\n{'='*60}")
    print("Verifying Installation")
    print(f"{'='*60}\n")
    
    python = get_venv_python(venv_path)
    
    # Create verification script
    verify_script = """
import sys
print(f"Python: {sys.version}")
print(f"Python path: {sys.executable}")
print()

# Check core packages
try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
except ImportError:
    print("❌ PyTorch not installed")
    sys.exit(1)

try:
    import torchvision
    print(f"✅ TorchVision: {torchvision.__version__}")
except ImportError:
    print("❌ TorchVision not installed")

try:
    import torchaudio
    print(f"✅ TorchAudio: {torchaudio.__version__}")
except ImportError:
    print("❌ TorchAudio not installed")

try:
    import librosa
    print(f"✅ Librosa: {librosa.__version__}")
except ImportError:
    print("❌ Librosa not installed")

try:
    import sklearn
    print(f"✅ Scikit-learn: {sklearn.__version__}")
except ImportError:
    print("❌ Scikit-learn not installed")

try:
    import gradio
    print(f"✅ Gradio: {gradio.__version__}")
except ImportError:
    print("❌ Gradio not installed")

try:
    import flask
    print(f"✅ Flask: {flask.__version__}")
except ImportError:
    print("❌ Flask not installed")

try:
    import psutil
    print(f"✅ psutil: {psutil.__version__}")
except ImportError:
    print("❌ psutil not installed")

try:
    import tensorboard
    print(f"✅ TensorBoard: installed")
except ImportError:
    print("❌ TensorBoard not installed")

print()
print("All core dependencies installed successfully!" if torch.cuda.is_available() or "--cpu" in sys.argv else "Installation complete (CPU mode)")
"""
    
    # Write and run verification script
    verify_path = Path("verify_install.py")
    verify_path.write_text(verify_script)
    
    try:
        result = run_command([str(python), str(verify_path)], check=False)
        if result.returncode != 0:
            print("\n⚠️  Some packages may not be installed correctly.")
    finally:
        verify_path.unlink(missing_ok=True)


def print_activation_instructions(venv_path):
    """Print instructions for activating the environment."""
    print(f"\n{'='*60}")
    print("Setup Complete!")
    print(f"{'='*60}\n")
    
    if platform.system() == "Windows":
        print("To activate the environment, run:")
        print(f"    {venv_path}\\Scripts\\activate")
        print()
        print("Or use:")
        print(f"    .\\{venv_path}\\Scripts\\Activate.ps1   # PowerShell")
        print(f"    {venv_path}\\Scripts\\activate.bat     # CMD")
    else:
        print("To activate the environment, run:")
        print(f"    source {venv_path}/bin/activate")
    
    print()
    print("Then you can run:")
    print("    python train.py")
    print("    python app.py")
    print("    python benchmark.py")
    print()
    print(f"Virtual environment location: {venv_path.absolute()}")


def main():
    parser = argparse.ArgumentParser(
        description="Setup MIMII Anomaly Detection environment"
    )
    parser.add_argument(
        "--venv", "-v",
        default="venv",
        help="Virtual environment name (default: venv)"
    )
    parser.add_argument(
        "--cpu", "-c",
        action="store_true",
        help="Install CPU-only PyTorch (no CUDA)"
    )
    parser.add_argument(
        "--skip-optional",
        action="store_true",
        help="Skip optional dependencies"
    )
    parser.add_argument(
        "--requirements", "-r",
        default="requirements.txt",
        help="Requirements file path"
    )
    
    args = parser.parse_args()
    
    print("🚀 MIMII Acoustic Anomaly Detection - Environment Setup")
    print(f"Platform: {platform.system()} {platform.machine()}")
    print(f"Python: {sys.version}")
    
    # Create virtual environment
    venv_path = create_virtual_environment(args.venv)
    
    # Upgrade pip
    upgrade_pip(venv_path)
    
    # Install PyTorch (CUDA or CPU)
    install_pytorch(venv_path, cpu_only=args.cpu)
    
    # Install requirements
    install_requirements(venv_path, args.requirements)
    
    # Install optional dependencies
    if not args.skip_optional:
        install_optional_dependencies(venv_path)
    
    # Verify installation
    verify_installation(venv_path, cpu_only=args.cpu)
    
    # Print activation instructions
    print_activation_instructions(venv_path)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
