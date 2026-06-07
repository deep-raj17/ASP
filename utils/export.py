"""
utils/export.py
────────────────────────────────────────────────────────
Model export utilities for production deployment.

Supports:
  - ONNX export with dynamic shapes
  - TorchScript export (tracing and scripting)
  - TensorRT conversion (if available)
  - Quantization (INT8) for edge deployment
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn

from config import Config
from models.hybrid_model import HybridAnomalyModel


def export_onnx(
    model: nn.Module,
    save_path: str,
    input_shape: Tuple[int, ...] = (1, 1, 128, 313),
    opset_version: int = 17,
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    simplify: bool = True,
) -> str:
    """
    Export model to ONNX format.

    Args:
        model: PyTorch model to export
        save_path: Path to save ONNX file
        input_shape: Example input shape (batch, channels, height, width)
        opset_version: ONNX opset version
        dynamic_axes: Dynamic axes for variable batch/frame sizes
        simplify: If True, run ONNX simplifier

    Returns:
        Path to exported ONNX file
    """
    model.eval()
    device = next(model.parameters()).device

    # Create dummy input
    dummy_input = torch.randn(input_shape, device=device)

    # Default dynamic axes for audio (batch and time dimensions)
    if dynamic_axes is None:
        dynamic_axes = {
            "input": {0: "batch_size", 3: "time_frames"},
            "embeddings": {0: "batch_size"},
            "logits": {0: "batch_size"},
            "reconstruction": {0: "batch_size", 3: "time_frames"},
        }

    # Export
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        input_names=["input"],
        output_names=["embeddings", "logits", "reconstruction", "attention_weights", "pooled_feat"],
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
        export_params=True,
    )

    print(f"[Export] ONNX model saved to {save_path}")

    # Simplify if requested
    if simplify:
        try:
            import onnx
            from onnxsim import simplify as onnx_simplify

            onnx_model = onnx.load(save_path)
            model_simp, check = onnx_simplify(onnx_model)

            if check:
                onnx.save(model_simp, save_path)
                print(f"[Export] ONNX model simplified successfully")
            else:
                warnings.warn("ONNX simplification failed, using original model")
        except ImportError:
            warnings.warn("onnx-simplifier not installed. Skipping simplification.")

    return save_path


def export_torchscript(
    model: nn.Module,
    save_path: str,
    method: str = "trace",
    input_shape: Tuple[int, ...] = (1, 1, 128, 313),
    optimize_for_mobile: bool = False,
) -> str:
    """
    Export model to TorchScript.

    Args:
        model: PyTorch model to export
        save_path: Path to save TorchScript file
        method: "trace" or "script"
        input_shape: Example input shape for tracing
        optimize_for_mobile: If True, apply mobile optimizations

    Returns:
        Path to exported TorchScript file
    """
    model.eval()
    device = next(model.parameters()).device

    if method == "trace":
        dummy_input = torch.randn(input_shape, device=device)
        scripted = torch.jit.trace(model, dummy_input)
    elif method == "script":
        scripted = torch.jit.script(model)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'trace' or 'script'.")

    # Optimization
    scripted = torch.jit.optimize_for_inference(scripted)

    if optimize_for_mobile:
        try:
            from torch.utils.mobile_optimizer import optimize_for_mobile
            scripted = optimize_for_mobile(scripted)
            print("[Export] Mobile optimizations applied")
        except ImportError:
            warnings.warn("Mobile optimizer not available")

    scripted.save(save_path)
    print(f"[Export] TorchScript model saved to {save_path}")

    return save_path


def quantize_model(
    model: nn.Module,
    calibration_loader,
    save_path: str,
    backend: str = "fbgemm",  # "fbgemm" for x86, "qnnpack" for ARM
) -> nn.Module:
    """
    Post-training static quantization for edge deployment.
    Reduces model size by ~4x and improves CPU inference speed.

    Args:
        model: Model to quantize
        calibration_loader: DataLoader for calibration
        save_path: Path to save quantized model
        backend: Quantization backend

    Returns:
        Quantized model
    """
    model.eval()
    model.cpu()

    # Fusion for Conv-BN-ReLU patterns
    torch.backends.quantized.engine = backend

    # Prepare model for quantization
    model.qconfig = torch.quantization.get_default_qconfig(backend)
    torch.quantization.prepare(model, inplace=True)

    # Calibration
    print("[Export] Calibrating quantization...")
    with torch.no_grad():
        for i, batch in enumerate(calibration_loader):
            if i >= 100:  # Calibrate on 100 batches max
                break
            mel = batch["mel"]
            model(mel)

    # Convert to quantized
    torch.quantization.convert(model, inplace=True)

    # Save
    torch.save(model.state_dict(), save_path)
    print(f"[Export] Quantized model saved to {save_path}")

    return model


def convert_to_tensorrt(
    onnx_path: str,
    save_path: str,
    fp16: bool = True,
    max_batch_size: int = 32,
    max_workspace_size_gb: float = 4.0,
) -> Optional[str]:
    """
    Convert ONNX model to TensorRT engine for maximum GPU performance.
    Requires TensorRT and pycuda to be installed.

    Args:
        onnx_path: Path to ONNX model
        save_path: Path to save TensorRT engine
        fp16: Enable FP16 precision
        max_batch_size: Maximum batch size
        max_workspace_size_gb: Maximum workspace size in GB

    Returns:
        Path to TensorRT engine or None if conversion failed
    """
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
    except ImportError:
        warnings.warn("TensorRT or pycuda not installed. Skipping TensorRT conversion.")
        return None

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    # Parse ONNX
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise RuntimeError("Failed to parse ONNX file")

    # Build config
    config = builder.create_builder_config()
    config.max_workspace_size = int(max_workspace_size_gb * (1 << 30))

    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # Build engine
    profile = builder.create_optimization_profile()
    profile.set_shape(
        "input",
        min=(1, 1, 128, 100),
        opt=(8, 1, 128, 313),
        max=(max_batch_size, 1, 128, 626),
    )
    config.add_optimization_profile(profile)

    print("[Export] Building TensorRT engine (this may take a while)...")
    engine = builder.build_engine(network, config)

    if engine is None:
        raise RuntimeError("Failed to build TensorRT engine")

    # Save engine
    with open(save_path, "wb") as f:
        f.write(engine.serialize())

    print(f"[Export] TensorRT engine saved to {save_path}")
    return save_path


class TensorRTRuntime:
    """TensorRT inference runtime."""

    def __init__(self, engine_path: str):
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
        except ImportError:
            raise ImportError("TensorRT and pycuda required")

        self.logger = trt.Logger(trt.Logger.WARNING)

        # Load engine
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        # Allocate buffers
        self._allocate_buffers()

    def _allocate_buffers(self):
        """Allocate GPU memory for inputs and outputs."""
        import numpy as np
        import pycuda.driver as cuda

        self.inputs = []
        self.outputs = []
        self.bindings = []

        for i in range(self.engine.num_bindings):
            shape = self.engine.get_binding_shape(i)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            size = trt.volume(shape)

            # Allocate host and device memory
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            self.bindings.append(int(device_mem))

            if self.engine.binding_is_input(i):
                self.inputs.append({"host": host_mem, "device": device_mem})
            else:
                self.outputs.append({"host": host_mem, "device": device_mem})

    def infer(self, input_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run inference on input tensor."""
        import pycuda.driver as cuda
        import numpy as np

        # Copy input to device
        input_array = input_tensor.cpu().numpy().ravel()
        np.copyto(self.inputs[0]["host"], input_array)
        cuda.memcpy_htod_async(self.inputs[0]["device"], self.inputs[0]["host"])

        # Execute
        self.context.execute_async_v2(bindings=self.bindings)

        # Copy outputs back
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out["host"], out["device"])

        cuda.Context.synchronize()

        # TODO: Parse outputs based on model structure
        return {}


def export_all_formats(
    model: nn.Module,
    cfg: Config,
    output_dir: str = "exports",
    input_shape: Tuple[int, ...] = (1, 1, 128, 313),
) -> Dict[str, str]:
    """
    Export model to all available formats.

    Returns:
        Dictionary mapping format names to file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    # PyTorch checkpoint
    pt_path = os.path.join(output_dir, "model.pt")
    torch.save(model.state_dict(), pt_path)
    results["pytorch"] = pt_path

    # ONNX
    try:
        onnx_path = os.path.join(output_dir, "model.onnx")
        export_onnx(model, onnx_path, input_shape)
        results["onnx"] = onnx_path

        # TensorRT (if available)
        try:
            trt_path = os.path.join(output_dir, "model.trt")
            trt_result = convert_to_tensorrt(onnx_path, trt_path)
            if trt_result:
                results["tensorrt"] = trt_result
        except Exception as e:
            warnings.warn(f"TensorRT conversion failed: {e}")
    except Exception as e:
        warnings.warn(f"ONNX export failed: {e}")

    # TorchScript
    try:
        ts_path = os.path.join(output_dir, "model.ts")
        export_torchscript(model, ts_path, method="trace", input_shape=input_shape)
        results["torchscript"] = ts_path
    except Exception as e:
        warnings.warn(f"TorchScript export failed: {e}")

    return results
