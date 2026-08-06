"""Check CUDA/GPU availability."""
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("No CUDA GPU detected. PyTorch will run on CPU.")
    print("To enable GPU: install PyTorch with CUDA support.")
    print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
