"""Verify split assignment and generate machine split table."""
import pandas as pd
import hashlib

df = pd.read_csv("metadata/dataset_manifest.csv")
print("=== SPLIT COUNTS ===")
print(df["split"].value_counts())
print()

print("=== MACHINE ID PER SPLIT ===")
for s in ["train", "val", "test"]:
    sdf = df[df["split"] == s]
    mids = sorted(sdf["machine_id"].unique())
    print(f"{s}: machine_ids={mids}, total_files={len(sdf)}")
    for mid in mids:
        sub = sdf[sdf["machine_id"] == mid]
        n_norm = (sub["label"] == "normal").sum()
        n_abn = (sub["label"] == "abnormal").sum()
        print(f"  {mid}: {len(sub)} files (normal={n_norm}, abnormal={n_abn})")
print()

print("=== MD5 BUCKET ASSIGNMENT (from dataset.py _split_name_for_path) ===")
bucket_map = {0: "train", 1: "val", 2: "test"}
for mid in ["id_00", "id_02", "id_04", "id_06"]:
    bucket = int(hashlib.md5(mid.encode()).hexdigest(), 16) % 3
    print(f"  {mid} -> bucket {bucket} ({bucket_map[bucket]})")
print()

# Check for any file-level randomness impacting split
rel_paths = df["relative_path"].unique()
print(f"Unique relative paths: {len(rel_paths)}")
print(f"Total rows: {len(df)}")
has_dup_paths = len(rel_paths) != len(df)
print(f"Duplicate file paths: {has_dup_paths}")

# Per-split label distribution
print()
print("=== LABEL DISTRIBUTION PER SPLIT ===")
for s in ["train", "val", "test"]:
    sdf = df[df["split"] == s]
    print(f"{s}: normal={(sdf['label']=='normal').sum()}, abnormal={(sdf['label']=='abnormal').sum()}")

# Check noise condition distribution
print()
print("=== NOISE CONDITION PER SPLIT ===")
for s in ["train", "val", "test"]:
    sdf = df[df["split"] == s]
    print(f"\n{s}:")
    for nc in sorted(sdf["noise_condition"].unique()):
        cnt = (sdf["noise_condition"] == nc).sum()
        print(f"  {nc}: {cnt}")

