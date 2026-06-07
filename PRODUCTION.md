# Production Deployment Guide

**Complete guide for deploying the MIMII Acoustic Anomaly Detection system in production environments with GPU optimization.**

---

## Quick Start

```bash
# 1. Train model
python train.py

# 2. Calibrate detector
python calibrate.py

# 3. Benchmark performance
python benchmark.py --batch-sizes 1 4 8 16 32

# 4. Deploy with Docker
python deploy.py deploy --version v1.0.0
```

---

## Production Features

### GPU Optimizations

| Feature | Implementation | Benefit |
|---------|----------------|---------|
| `torch.compile()` | `utils/gpu_utils.py` | 20-30% speedup on PyTorch 2.x |
| TF32 Matmul | Automatic on Ampere+ | 2x faster FP32 ops |
| Mixed Precision | FP16/BF16 training | 2x throughput, half memory |
| CUDA Graphs | In `ProductionDetector` | Reduced CPU overhead |
| TensorRT | `utils/export.py` | Maximum inference throughput |

### Production Infrastructure

| Component | File | Purpose |
|-----------|------|---------|
| Production Detector | `inference/production_detector.py` | Optimized batch inference |
| Input Validation | `utils/validation.py` | Safe error handling |
| Monitoring | `utils/monitoring.py` | Metrics & alerting |
| Data Pipeline | `utils/data_pipeline.py` | Cached streaming loader |
| REST API | `production_api.py` | Production HTTP API |
| Deployment | `deploy.py` | Automated deployment |

---

## Deployment Options

### Option 1: Docker (Recommended)

```bash
# Build and run
python deploy.py deploy --version v1.0.0

# Or manually:
docker-compose up -d
```

### Option 2: Direct Python

```bash
# Setup environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run production API
python production_api.py --port 5000

# Or run Gradio UI
python app.py
```

### Option 3: Cloud (Hugging Face Spaces)

```bash
# Push to HF Spaces (GPU enabled)
git push huggingface main
```

---

## Performance Benchmarking

```bash
# Full benchmark suite
python benchmark.py \
    --batch-sizes 1 4 8 16 32 \
    --num-runs 100 \
    --test-compilation \
    --output results.txt
```

### Expected Performance (RTX 4070)

| Batch Size | Latency (ms) | Throughput (samples/sec) |
|------------|--------------|--------------------------|
| 1          | 45-55        | 18-22                    |
| 4          | 120-150      | 26-33                    |
| 8          | 200-250      | 32-40                    |
| 16         | 380-450      | 35-42                    |
| 32         | 700-850      | 37-45                    |

### With `torch.compile()` (PyTorch 2.x)

| Batch Size | Latency (ms) | Speedup |
|------------|--------------|---------|
| 1          | 35-42        | 1.3x    |
| 8          | 160-190      | 1.4x    |
| 32         | 550-650      | 1.5x    |

---

## Model Export

```python
from utils.export import export_all_formats
from models.hybrid_model import HybridAnomalyModel

model = HybridAnomalyModel(cfg.model)
# Load checkpoint...

# Export to all formats
results = export_all_formats(
    model, cfg,
    output_dir="exports",
    input_shape=(1, 1, 128, 313)
)

# Results:
# - exports/model.pt (PyTorch)
# - exports/model.onnx (ONNX)
# - exports/model.ts (TorchScript)
# - exports/model.trt (TensorRT)
```

### TensorRT Optimization (Maximum Performance)

```bash
# Requires: pip install tensorrt pycuda

python -c "
from utils.export import convert_to_tensorrt
convert_to_tensorrt(
    'exports/model.onnx',
    'exports/model.trt',
    fp16=True,
    max_batch_size=32
)
"
```

---

## API Usage

### REST API Endpoints

```bash
# Health check
curl http://localhost:5000/health

# Single prediction
curl -X POST -F "file=@sample.wav" http://localhost:5000/predict

# Batch prediction
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"files": ["file1.wav", "file2.wav"]}' \
  http://localhost:5000/predict/batch

# Async batch job
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"files": ["file1.wav", "file2.wav"]}' \
  http://localhost:5000/predict/async

# Check job status
curl http://localhost:5000/jobs/job_1234567890

# Get metrics
curl http://localhost:5000/metrics
```

### Python Client

```python
import requests

# Single inference
with open("audio.wav", "rb") as f:
    response = requests.post(
        "http://localhost:5000/predict",
        files={"file": f}
    )
result = response.json()
print(f"Anomaly score: {result['result']['anomaly_score']}")
print(f"Risk level: {result['result']['system']['risk_level']}")
```

---

## Monitoring & Observability

### Prometheus Metrics

```bash
# Metrics endpoint
curl http://localhost:5000/metrics

# Prometheus-compatible output:
# mimii_inference_duration_ms{status="success"} 45.2
# mimii_gpu_memory_allocated_gb 2.34
# mimii_inference_anomaly_score 0.82
```

### Structured Logging

```python
from utils.validation import setup_logging

logger = setup_logging(
    log_file="logs/production.log",
    level=logging.INFO
)

logger.info("Processing file", extra={"file": "audio.wav"})
logger.error("Inference failed", extra={"error": str(e)})
```

### Performance Tracking

```python
from utils.monitoring import timed, record_metric

# Automatic timing
with timed("inference", batch_size=8):
    results = detector.detect_batch(batch)

# Manual recording
record_metric("custom_metric", value=42.0, source="my_module")

# Get statistics
from utils.monitoring import get_metrics
stats = get_metrics()
```

---

## Configuration for Production

### `config.py` Production Settings

```python
@dataclass
class TrainingConfig:
    # GPU Optimization
    device: str = "cuda"
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 4

    # Data Loading
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2

    # Memory
    batch_size: int = 32  # Adjust based on GPU memory

@dataclass
class InferenceConfig:
    # Score weights (must sum to 1.0)
    w_recon: float = 0.30
    w_embed: float = 0.25
    w_mahal: float = 0.30
    w_contra: float = 0.15

    # Thresholds
    percentile_threshold: float = 95.0
```

---

## Troubleshooting

### GPU Memory Issues

```python
# Reduce batch size
python train.py  # Edit config.py: batch_size = 16

# Enable gradient checkpointing
# In model definition: torch.utils.checkpoint.checkpoint()

# Use CPU offloading
# In config.py: device = "cpu" (slower but works)
```

### CUDA Out of Memory

```bash
# Clear cache
python -c "from utils.gpu_utils import empty_cache; empty_cache()"

# Monitor memory
python -c "
from utils.gpu_utils import print_memory_stats
print_memory_stats('debug')
"
```

### Slow Inference

```bash
# Check if torch.compile() is working
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'torch.compile available: {hasattr(torch, \"compile\")}')
"

# Run benchmark
python benchmark.py --test-compilation
```

---

## Production Checklist

- [ ] Model trained and validated
- [ ] Detector calibrated on normal data
- [ ] Benchmarked on target hardware
- [ ] Exported to production format (ONNX/TorchScript)
- [ ] Docker image built and tested
- [ ] Health checks passing
- [ ] Monitoring configured
- [ ] Alerts configured
- [ ] Backup/rollback plan documented
- [ ] Load testing completed

---

## Support

- **Issues**: Check `logs/` directory for detailed error logs
- **Performance**: Run `python benchmark.py` for diagnostics
- **Health**: Use `python deploy.py health` for container checks
- **GPU**: Verify with `python -c "import torch; print(torch.cuda.is_available())"`
