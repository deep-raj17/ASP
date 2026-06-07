#!/usr/bin/env python3
"""Quick smoke test: env, GPU, artifacts, and one forward pass (no training)."""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

def main() -> int:
    import torch
    from config import cfg
    from paths import PATHS, resolve_model_checkpoint, artifacts_status
    from models.hybrid_model import HybridAnomalyModel
    from utils.checkpoint import load_model_weights

    print("=" * 60)
    print("  MIMII Inference Verification (no training)")
    print("=" * 60)

    status = artifacts_status()
    for k, v in status.items():
        print(f"  {k}: {v}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  device: {device}")
    if device.type == "cuda":
        print(f"  gpu: {torch.cuda.get_device_name(0)}")

    ckpt, prec = resolve_model_checkpoint()
    if not os.path.isfile(ckpt):
        print(f"\nERROR: Missing weights at {ckpt}")
        return 1

    calib = PATHS.detector_calibration
    if not os.path.isfile(calib):
        print(f"\nWARN: Missing calibration at {calib}")

    model = HybridAnomalyModel(cfg.model).to(device)
    epoch, loaded_prec = load_model_weights(model, ckpt, device)
    model.eval()
    print(f"\n  loaded: {ckpt}")
    print(f"  precision: {loaded_prec} (resolved: {prec}), epoch={epoch}")

    dummy = torch.randn(1, 1, cfg.data.n_mels, 313, device=device)
    with torch.inference_mode():
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = model(dummy)
        else:
            out = model(dummy)
    print(f"  forward OK, logits shape: {out['logits'].shape}")
    print("\n  All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
