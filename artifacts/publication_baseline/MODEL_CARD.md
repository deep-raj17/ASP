# CHAAD Model Card — PMPS-01 Snapshot

## Model

CHAAD is a supervised hybrid acoustic anomaly detector using an
EfficientNet-B4 spectrogram backbone, Transformer temporal encoder, attention
pooling, classifier, contrastive projection, and reconstruction branch.

## Inputs and outputs

Input audio is configured for 16 kHz, 10-second, multi-channel MIMII WAV data
and converted to normalized log-mel features. The audited validation export
contains a continuous anomaly probability.

## Training objective

Weighted BCE classification + supervised contrastive loss + reconstruction
loss. EXP-CHAAD-001 selected epoch 6 by minimum validation loss.

## Intended use

Research on machine-independent industrial acoustic anomaly detection.

## Out-of-scope use

Safety-critical autonomous shutdown, clinical use, surveillance, and claims of
cross-dataset or real-world factory generalization without additional evidence.

## Hardware and dependencies

Python/PyTorch; GPU optional for inference. Exact versions are in
`environment_full.json`.

## Limitations and known issues

Validation ROC-AUC is approximately 0.60026 after prediction-export correction;
the model is underfit; there is no authorized held-out test result in this
audit; multi-seed and cross-platform reproducibility are not established.

## Ethical considerations

False alarms and missed failures can affect worker safety and maintenance cost.
Human oversight and site-specific validation are required.
