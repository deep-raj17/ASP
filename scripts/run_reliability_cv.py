#!/usr/bin/env python3
"""Grouped out-of-fold validation runner for reliability-aware fusion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset

from models.reliability import (
    MetadataMapper,
    ReliabilityGate,
    ReliabilityGateTrainer,
)
from utils.experiment_contract import assert_split_access
from utils.seed import set_seed


SCORE_COLUMNS = ["recon_error", "embed_dist", "mahal_dist", "contra_dist"]


class ReliabilityFrameDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        mapper: MetadataMapper,
        embedding_columns: list[str],
    ):
        self.frame = frame.reset_index(drop=True)
        self.mapper = mapper
        self.embedding_columns = embedding_columns

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict:
        row = self.frame.iloc[index]
        machine = str(row["machine_type"])
        noise = str(row["noise_condition"])
        return {
            "calibrated_scores": torch.tensor(
                row[SCORE_COLUMNS].to_numpy(dtype=np.float32),
                dtype=torch.float32,
            ),
            "embedding": torch.tensor(
                row[self.embedding_columns].to_numpy(dtype=np.float32),
                dtype=torch.float32,
            ),
            "machine_idx": torch.tensor(
                self.mapper.map_machine(machine),
                dtype=torch.long,
            ),
            "noise_idx": torch.tensor(
                self.mapper.map_noise(noise),
                dtype=torch.long,
            ),
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
            "machine": machine,
            "snr": noise,
            "sample_id": str(row["sample_id"]),
        }


def validate_feature_frame(frame: pd.DataFrame) -> list[str]:
    embedding_columns = sorted(
        (column for column in frame.columns if column.startswith("embedding_")),
        key=lambda value: int(value.split("_")[-1]),
    )
    required = set(SCORE_COLUMNS) | {
        "sample_id",
        "label",
        "machine_type",
        "machine_id",
        "noise_condition",
        "source_recording",
        "split",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Reliability feature frame missing columns: {missing}")
    if not embedding_columns:
        raise ValueError("Reliability feature frame has no embedding columns")
    normalized_splits = set(
        frame["split"].astype(str).str.lower().replace({"val": "validation"})
    )
    if normalized_splits != {"validation"}:
        raise ValueError(f"Reliability CV requires validation-only rows: {normalized_splits}")
    if frame["sample_id"].astype(str).duplicated().any():
        raise ValueError("Reliability feature frame contains duplicate sample IDs")
    numeric = frame[SCORE_COLUMNS + embedding_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )
    if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        raise ValueError("Reliability feature frame contains non-finite values")
    return embedding_columns


def run_grouped_reliability_cv(
    frame: pd.DataFrame,
    *,
    seed: int,
    outer_splits: int = 5,
    inner_splits: int = 4,
    epochs: int = 50,
    patience: int = 10,
    batch_size: int = 256,
    device: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    embedding_columns = validate_feature_frame(frame)
    set_seed(seed, deterministic_cudnn=True)
    target_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    labels = (frame["label"].to_numpy(dtype=float) >= 0.5).astype(int)
    groups = (
        frame["machine_type"].astype(str)
        + "|"
        + frame["machine_id"].astype(str)
        + "|"
        + frame["source_recording"].astype(str)
    ).to_numpy()
    outer = StratifiedGroupKFold(
        n_splits=outer_splits,
        shuffle=True,
        random_state=seed,
    )
    predictions = np.full(len(frame), np.nan, dtype=np.float64)
    fold_assignments = np.full(len(frame), -1, dtype=int)
    fold_records = []

    for fold, (outer_train_idx, holdout_idx) in enumerate(
        outer.split(frame, labels, groups)
    ):
        outer_train = frame.iloc[outer_train_idx].reset_index(drop=True)
        outer_labels = labels[outer_train_idx]
        outer_groups = groups[outer_train_idx]
        inner = StratifiedGroupKFold(
            n_splits=inner_splits,
            shuffle=True,
            random_state=seed + fold,
        )
        gate_train_rel, gate_select_rel = next(
            inner.split(outer_train, outer_labels, outer_groups)
        )
        gate_train = outer_train.iloc[gate_train_rel]
        gate_select = outer_train.iloc[gate_select_rel]
        holdout = frame.iloc[holdout_idx]

        mapper = MetadataMapper().fit(
            gate_train["machine_type"].astype(str).tolist(),
            gate_train["noise_condition"].astype(str).tolist(),
        )
        train_dataset = ReliabilityFrameDataset(
            gate_train,
            mapper,
            embedding_columns,
        )
        select_dataset = ReliabilityFrameDataset(
            gate_select,
            mapper,
            embedding_columns,
        )
        holdout_dataset = ReliabilityFrameDataset(
            holdout,
            mapper,
            embedding_columns,
        )
        generator = torch.Generator().manual_seed(seed + fold)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
        )
        select_loader = DataLoader(
            select_dataset,
            batch_size=batch_size,
            shuffle=False,
        )
        holdout_loader = DataLoader(
            holdout_dataset,
            batch_size=batch_size,
            shuffle=False,
        )
        gate = ReliabilityGate(
            embedding_dim=len(embedding_columns),
            num_scores=len(SCORE_COLUMNS),
            num_machine_types=mapper.num_machines,
            num_noise_conditions=mapper.num_noises,
        ).to(target_device)
        trainer = ReliabilityGateTrainer(gate)
        result = trainer.fit(
            train_loader,
            evaluation_loader=select_loader,
            epochs=epochs,
            patience=patience,
            verbose=False,
        )

        gate.eval()
        fold_scores = []
        with torch.inference_mode():
            for batch in holdout_loader:
                fused, _ = gate(
                    batch["calibrated_scores"].to(target_device),
                    batch["embedding"].to(target_device),
                    batch["machine_idx"].to(target_device),
                    batch["noise_idx"].to(target_device),
                )
                fold_scores.extend(fused.cpu().numpy().tolist())
        predictions[holdout_idx] = np.asarray(fold_scores)
        fold_assignments[holdout_idx] = fold
        fold_records.append(
            {
                "fold": fold,
                "train_rows": len(gate_train),
                "selection_rows": len(gate_select),
                "holdout_rows": len(holdout),
                "best_epoch": result.best_epoch,
                "best_loss": result.best_loss,
            }
        )

    if not np.isfinite(predictions).all() or (fold_assignments < 0).any():
        raise RuntimeError("Reliability CV failed to produce complete finite OOF predictions")
    output = frame[["sample_id", "label", "split"]].copy()
    output["predicted_score"] = predictions
    output["fold"] = fold_assignments
    return output, {
        "seed": seed,
        "outer_splits": outer_splits,
        "inner_splits": inner_splits,
        "folds": fold_records,
        "status": "PASS",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--predictions-output", required=True)
    parser.add_argument("--report-output", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--phase", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    assert_split_access(phase=args.phase, split="validation")
    frame = pd.read_csv(args.features)
    predictions, report = run_grouped_reliability_cv(
        frame,
        seed=args.seed,
        epochs=args.epochs,
        patience=args.patience,
        device=args.device,
    )
    predictions_path = Path(args.predictions_output)
    report_path = Path(args.report_output)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with predictions_path.open("x", encoding="utf-8", newline="") as handle:
        predictions.to_csv(handle, index=False)
    with report_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
