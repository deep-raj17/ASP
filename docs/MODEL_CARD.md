# Model Card — HybridAnomalyModel

> CHAAD Project

## Model Identity

| Field | Value |
|-------|-------|
| Model name | HybridAnomalyModel |
| Version | Current (no version tag) |
| Framework | PyTorch 2.x |
| File | `models/hybrid_model.py` |

## Intended Use

Detect anomalous sounds from industrial machinery (fans, pumps, sliders, valves) by processing 10-second mono audio segments at 16kHz. Produces a continuous anomaly score (0-1) and binary anomaly classification.

## Out-of-Scope Use

- Unsupervised detection (model is supervised)
- Non-acoustic sensor data
- Non-industrial sounds (speech, music, environmental)
- Real-time streaming without buffering to 10-second windows
- Machines not represented in MIMII training data

## Architecture

```text
Input: mel spectrogram (B, 1, 128, T)
  │
  ├─ Autoencoder Branch ──→ reconstruction (B, 1, 128, T)  ──→ recon error
  │
  └─ CNN Backbone (EfficientNet-B4, 1-channel) ──→ spatial features (B, C, H, W)
       │
       └─ Flatten → (B, T_seq, C) → Linear proj → (B, T_seq, 256)
            │
            └─ Transformer Encoder (4 layers, 8 heads) → (B, T_seq, 256)
                 │
                 └─ Attention Pooling → (B, 256)
                      │
                      ├─ Classifier → logits (B, 1)
                      └─ Projector → L2-norm embeddings (B, 256)
```

Components:
- **Backbone**: EfficientNet-B4, first conv patched from 3→1 channel
- **Temporal**: Transformer Encoder (default) or BiLSTM
- **Attention Pool**: Soft-attention over temporal dimension
- **Classifier**: 2-layer MLP → binary logit
- **Projector**: 3-layer MLP → L2-normalized 256-dim embedding
- **Autoencoder**: 3-layer Conv2D encoder/decoder → reconstruction

## Inputs

| Field | Shape | Description |
|-------|-------|-------------|
| mel | (B, 1, 128, T) | Log-mel spectrogram, normalized to [0,1] |

## Outputs

| Key | Shape | Description |
|-----|-------|-------------|
| embeddings | (B, 256) | L2-normalized feature vectors |
| logits | (B, 1) | Binary classification logit |
| reconstruction | (B, 1, 128, T) | Autoencoder output |
| attention_weights | (B, T_seq, 1) | Temporal attention weights |
| pooled_feat | (B, 256) | Pre-projection pooled features |

## Anomaly Signals

Four complementary signals are computed from model outputs (in `inference/detector.py`):

| Signal | Source | Normalization |
|--------|--------|---------------|
| Reconstruction error | MSE(input, reconstruction) | z-score |
| Embedding distance | 1 - cosine_sim(embedding, ref_mean) | z-score |
| Mahalanobis distance | sqrt((x-μ)ᵀ Σ⁻¹ (x-μ)) | z-score |
| Contrastive distance | 1 - mean(top-k cosine sim to ref_pool) | z-score |

## Fusion Method

- **Current (baseline)**: Fixed hand-selected weights (0.30, 0.25, 0.30, 0.15)
- **Novel (proposed)**: Reliability-aware learned fusion (`models/reliability.py`)

## Training Protocol

| Parameter | Value | Source |
|-----------|-------|--------|
| Loss | BCE + SupCon + MSE Reconstruction | `training/loss.py` |
| Optimizer | AdamW (lr=1e-4, wd=1e-4) | `config.py` |
| Batch size | 32 (effective=64 with grad accum=2) | `config.py` |
| Epochs | 100 | `config.py` |
| Scheduler | OneCycleLR | `config.py` |
| Mixed precision | FP16 via torch.cuda.amp | `config.py` |
| BCE pos_weight | 5.0 (class imbalance) | `config.py` |
| Loss weights | BCE=1.0, Contrastive=0.3, Recon=0.05 | `config.py` |
| Seed | 42 | `config.py` (new) |

## Calibration

All calibration statistics (μ, σ per signal, reference mean, covariance, reference pool) are computed on **train_normal only** via `AnomalyDetector.fit_reference_distribution()`.

## Thresholding

- **Method**: Youden's J statistic on validation split
- **Storage**: `artifacts/threshold_metadata.json`
- **Test evaluation**: Frozen threshold from validation

## Performance Status

**FINAL PERFORMANCE IS UNVERIFIED.** No model checkpoint exists under the current machine-independent 3-split protocol. Legacy metrics (ROC-AUC 99.99997%) are from an unknown split configuration and cannot be cited.

## Limitations

1. **Supervised only**: Requires labeled abnormal samples for training
2. **Fixed duration**: Assumes exactly 10-second audio segments
3. **Single microphone**: One audio channel only
4. **Known machine types**: Trained on 4 MIMII machine types — generalization to others is UNKNOWN
5. **Stationary noise**: Additive noise model may not capture real factory conditions
6. **No uncertainty quantification**: Single-point anomaly score without confidence intervals

## Ethical Considerations

- Intended for industrial equipment monitoring, not surveillance or personnel monitoring
- False negatives could result in missed equipment failures — model should supplement, not replace, human inspection
- Dataset bias: 4.36:1 normal-to-abnormal ratio may affect precision/recall tradeoffs

## Deployment

- **ONNX export**: Supported via `utils/export.py`
- **Docker**: `Dockerfile` and `docker-compose.yml` available
- **Edge**: Raspberry Pi deployment via `edge_deploy/`
- **Production API**: `production_api.py`

## Checkpoint Status

| File | Status |
|------|--------|
| `checkpoints/best_model.pt` | UNVERIFIED (may not exist) |
| `checkpoints/epoch_*.pt` | UNVERIFIED |
| `checkpoints/detector_calibration.pt` | UNVERIFIED |

---

*Metrics and performance claims are UNVERIFIED pending independent test-set evaluation under the current split protocol. See `docs/METRICS_REGISTRY.md` for metric definitions.*
