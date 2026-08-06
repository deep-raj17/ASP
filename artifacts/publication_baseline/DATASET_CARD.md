# MIMII Dataset Card — PMPS-01 Snapshot

## Identity

- Source: MIMII research dataset
- Local root: `E:\MIMII`
- Version: UNKNOWN
- License: UNKNOWN; must be confirmed before redistribution/publication
- Samples: 53,046
- Manifest SHA-256: `7c689508cbed4d49d05ec2891b315b27722ff01a8a62b6b1c4f610e3afcd0136`

## Composition and split protocol

Four machine types (fan, pump, slider, valve), four physical machine IDs, and
normal/abnormal labels. Machine-independent split: train=id_04,
validation=id_00+id_02, test=id_06.

## Known limitations

Class imbalance, four machine families, limited machine-ID diversity, fixed
noise conditions, unknown local dataset version/license, and incomplete live
readability/NaN verification of the full 135.8 GB corpus.

## Integrity status

Manifest/file existence and size checks pass. Full-corpus live decoding,
finite-value scanning, and current-file hash recomputation remain UNVERIFIED.
