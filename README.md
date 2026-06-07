# MIMII Industrial Acoustic Anomaly Detection

Production-oriented predictive maintenance system for machine-sound anomaly
detection. The default workflow is inference-only: it loads the trained MIMII
model, calibration statistics, and exported edge artifacts already present in
this repository.

## Quick Start

Windows PowerShell:

```powershell
.\setup.bat
.\launch_ui.bat
```

The UI starts on `http://127.0.0.1:7860` or the next available port. The model
loads lazily on the first analysis request so startup stays responsive.

For the REST API:

```cmd
set APP_USE_TORCH_COMPILE=0
python production_api.py --port 5000
```

## Frozen Production Artifacts

Do not regenerate these unless you intentionally retrain and recalibrate.

| Role | Path |
|---|---|
| Primary PyTorch weights | `checkpoints/best_model.pt` |
| Detector calibration | `checkpoints/detector_calibration.pt` |
| GPU-friendly FP16-stored weights | `artifacts/models/best_model_fp16.pt` |
| Edge INT8 ONNX classifier | `edge_deploy/models/classifier_int8.onnx` |
| Packaging mirror for INT8 ONNX | `artifacts/onnx_int8/classifier_int8.onnx` |
| Artifact metadata | `artifacts/models/manifest.json` |

GPU inference loads `artifacts/models/best_model_fp16.pt` when CUDA is
available, otherwise `checkpoints/best_model.pt`. Runtime scoring remains in
FP32 while CUDA forward passes use autocast FP16 for speed.

## Project Layout

```text
ASP/
  app.py                         Gradio inference UI
  production_api.py              Flask REST API
  paths.py                       Canonical artifact paths and status
  config.py                      Dataset, model, training, inference config
  scripts/verify_inference.py    Smoke test for weights and one forward pass
  checkpoints/                   Frozen PyTorch weights and calibration
  artifacts/                     Derived model artifacts and manifest
  edge_deploy/                   ONNX Runtime / Raspberry Pi deployment
  models/                        Hybrid CNN-temporal-autoencoder model
  inference/                     Calibrated detector and batch detector
  training/                      Optional training loop and losses
  utils/                         Audio, metrics, GPU, validation helpers
```

Generated folders such as `venv/`, `__pycache__/`, and `logs/` are not part of
the source project. Recreate `venv/` with `setup.bat` when needed.

## Requirements

| Component | Version |
|---|---|
| Python | 3.11+ |
| PyTorch | 2.1+ |
| CUDA | Optional, CUDA 12.1 wheels via `setup.bat` |
| GPU | Recommended for low-latency inference |

## Dataset Layout for Optional Training

Set `cfg.data.dataset_dir` in `config.py` to the local MIMII root:

```text
MIMII_DATASET/
  0_dB_fan/
    fan/
      id_00/
        normal/*.wav
        abnormal/*.wav
  0_dB_pump/
    pump/...
  6_dB_fan/
    fan/...
```

## Optional Training Pipeline

Use this only when you want new weights.

```cmd
python verify_dataset.py
python train.py
python calibrate.py
python evaluate.py
```

After training, remove intermediate epoch checkpoints while preserving the
production artifacts:

```cmd
python cleanup.py
```

To also remove the local virtual environment and rebuild it later:

```cmd
python cleanup.py --remove-venv
```

## Model Architecture

| Component | Details |
|---|---|
| CNN backbone | EfficientNet-B4 by default, patched for 1-channel mel input |
| Temporal module | Transformer encoder or BiLSTM |
| Pooling | Learned attention pooling |
| Embedding head | L2-normalized contrastive embedding |
| Classifier head | Binary anomaly logit |
| Reconstruction branch | Convolutional autoencoder |

## Anomaly Scoring

The detector fuses four calibrated signals:

| Signal | Weight |
|---|---|
| Reconstruction error | 30% |
| Embedding distance | 25% |
| Mahalanobis distance | 30% |
| Contrastive nearest-neighbor score | 15% |

Each signal is z-score normalized against calibration statistics, sigmoid
mapped, and fused into a 0-1 anomaly score.

## Validation

Run the inference smoke test:

```cmd
python scripts\verify_inference.py
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `venv\Scripts\python.exe` fails | Delete `venv/` and run `setup.bat` |
| Dataset not found | Set `cfg.data.dataset_dir` in `config.py` |
| CUDA out of memory | Reduce `cfg.training.batch_size` |
| No WAV files found | Check the MIMII folder layout |
| Worker errors on Windows | Keep `cfg.training.num_workers = 0` |
| Port already in use | The UI auto-selects the next free port |

## License

Research and educational use. The MIMII dataset has its own license terms.
# ASP
# ASP
