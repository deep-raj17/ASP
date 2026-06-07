"""
edge_deploy - Raspberry Pi 5 Production Deployment Pipeline
────────────────────────────────────────────────────────────
PyTorch → ONNX → INT8 quantization + real-time streaming
inference engine for industrial acoustic anomaly detection.

Modules:
    edge_quantize  : Model export + INT8 static PTQ pipeline
    edge_runtime   : ONNX Runtime ARM64 inference wrapper
    edge_streaming : Real-time ring-buffer audio processing
    edge_watchdog  : Industrial watchdog + health monitoring
    edge_benchmark : Latency/thermal/accuracy benchmarking
"""

__version__ = "1.0.0"
