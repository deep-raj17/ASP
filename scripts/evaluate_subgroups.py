"""
scripts/evaluate_subgroups.py
────────────────────────────────────────────────────────
Evaluate performance per machine type, machine ID, and noise condition.

Usage:
    python scripts/evaluate_subgroups.py
"""

import pandas as pd
import json
from pathlib import Path


def evaluate_subgroups():
    """Evaluate subgroup performance from dataset manifest."""
    print("=" * 60)
    print("SUBGROUP EVALUATION")
    print("=" * 60)
    
    # Load manifest
    manifest_path = "metadata/dataset_manifest.csv"
    if not Path(manifest_path).exists():
        print(f"Manifest not found: {manifest_path}")
        print("Run generate_dataset_manifest.py first")
        return
    
    df = pd.read_csv(manifest_path)
    
    # Per-machine-type results
    print("\n--- Per Machine Type ---")
    machine_type_results = []
    for machine_type in df['machine_type'].unique():
        subset = df[df['machine_type'] == machine_type]
        for split in subset['split'].unique():
            split_subset = subset[subset['split'] == split]
            result = {
                "machine_type": machine_type,
                "split": split,
                "total_samples": len(split_subset),
                "normal_count": (split_subset['label'] == 'normal').sum(),
                "abnormal_count": (split_subset['label'] == 'abnormal').sum(),
                "machine_ids": split_subset['machine_id'].unique().tolist()
            }
            machine_type_results.append(result)
            print(f"{machine_type} ({split}): {len(split_subset)} samples "
                  f"(normal={result['normal_count']}, abnormal={result['abnormal_count']})")
    
    # Per-machine-ID results
    print("\n--- Per Machine ID ---")
    machine_id_results = []
    for machine_id in df['machine_id'].unique():
        subset = df[df['machine_id'] == machine_id]
        for split in subset['split'].unique():
            split_subset = subset[subset['split'] == split]
            result = {
                "machine_id": machine_id,
                "split": split,
                "total_samples": len(split_subset),
                "normal_count": (split_subset['label'] == 'normal').sum(),
                "abnormal_count": (split_subset['label'] == 'abnormal').sum(),
                "machine_types": split_subset['machine_type'].unique().tolist()
            }
            machine_id_results.append(result)
            print(f"{machine_id} ({split}): {len(split_subset)} samples "
                  f"(normal={result['normal_count']}, abnormal={result['abnormal_count']})")
    
    # Per-noise-condition results
    print("\n--- Per Noise Condition ---")
    noise_condition_results = []
    for noise_condition in df['noise_condition'].unique():
        subset = df[df['noise_condition'] == noise_condition]
        for split in subset['split'].unique():
            split_subset = subset[subset['split'] == split]
            result = {
                "noise_condition": noise_condition,
                "split": split,
                "total_samples": len(split_subset),
                "normal_count": (split_subset['label'] == 'normal').sum(),
                "abnormal_count": (split_subset['label'] == 'abnormal').sum()
            }
            noise_condition_results.append(result)
            print(f"{noise_condition} ({split}): {len(split_subset)} samples "
                  f"(normal={result['normal_count']}, abnormal={result['abnormal_count']})")
    
    # Save results
    Path("reports").mkdir(parents=True, exist_ok=True)
    
    pd.DataFrame(machine_type_results).to_csv("reports/per_machine_results.csv", index=False)
    pd.DataFrame(machine_id_results).to_csv("reports/per_machine_id_results.csv", index=False)
    pd.DataFrame(noise_condition_results).to_csv("reports/per_noise_condition_results.csv", index=False)
    
    print("\n" + "=" * 60)
    print("SUBGROUP RESULTS SAVED")
    print("=" * 60)
    print("  - reports/per_machine_results.csv")
    print("  - reports/per_machine_id_results.csv")
    print("  - reports/per_noise_condition_results.csv")
    
    # Note about metrics
    print("\nNOTE: Performance metrics (ROC-AUC, etc.) require model predictions.")
    print("These tables show sample counts per subgroup only.")
    print("To compute actual metrics per subgroup, run with a trained model.")


if __name__ == "__main__":
    evaluate_subgroups()
