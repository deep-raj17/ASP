"""
scripts/run_diagnostic_baselines.py
────────────────────────────────────────────────────────
Run diagnostic baselines under the same split and preprocessing protocol.

Usage:
    python scripts/run_diagnostic_baselines.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, balanced_accuracy_score, f1_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import cfg
from data.dataset import MIMIIDataset


def majority_class_baseline(y_true):
    """Majority class baseline."""
    majority_class = np.bincount(y_true.astype(int)).argmax()
    y_pred = np.full_like(y_true, majority_class)
    
    # For AUC, need scores - use constant score
    y_scores = np.full_like(y_true, 0.5, dtype=float)
    
    metrics = {
        "roc_auc": 0.5,  # Random for constant scores
        "pr_auc": np.mean(y_true),  # PR-AUC = proportion of positives
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "majority_class": int(majority_class)
    }
    
    return metrics


def random_score_baseline(y_true, n_samples=1000):
    """Random score baseline."""
    y_scores = np.random.rand(len(y_true))
    y_pred = (y_scores > 0.5).astype(int)
    
    metrics = {
        "roc_auc": float(roc_auc_score(y_true, y_scores)),
        "pr_auc": float(average_precision_score(y_true, y_scores)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0))
    }
    
    return metrics


def logistic_regression_baseline(X_train, y_train, X_val, y_val):
    """Logistic regression on flattened mel spectrograms."""
    print("Training Logistic Regression...")
    
    # Convert labels to int
    y_train = y_train.astype(int)
    y_val = y_val.astype(int)
    
    # Flatten spectrograms
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)
    
    # Use subset for efficiency
    subset_size = min(1000, len(X_train_flat))
    indices = np.random.choice(len(X_train_flat), subset_size, replace=False)
    X_train_subset = X_train_flat[indices]
    y_train_subset = y_train[indices]
    
    clf = LogisticRegression(max_iter=1000, n_jobs=-1)
    clf.fit(X_train_subset, y_train_subset)
    
    y_scores = clf.predict_proba(X_val_flat)[:, 1]
    y_pred = (y_scores > 0.5).astype(int)
    
    metrics = {
        "roc_auc": float(roc_auc_score(y_val, y_scores)),
        "pr_auc": float(average_precision_score(y_val, y_scores)),
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_val, y_pred)),
        "f1": float(f1_score(y_val, y_pred, zero_division=0))
    }
    
    return metrics


def random_forest_baseline(X_train, y_train, X_val, y_val):
    """Random forest on flattened mel spectrograms."""
    print("Training Random Forest...")
    
    # Convert labels to int
    y_train = y_train.astype(int)
    y_val = y_val.astype(int)
    
    # Flatten spectrograms
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)
    
    # Use subset for efficiency
    subset_size = min(1000, len(X_train_flat))
    indices = np.random.choice(len(X_train_flat), subset_size, replace=False)
    X_train_subset = X_train_flat[indices]
    y_train_subset = y_train[indices]
    
    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, max_depth=10)
    clf.fit(X_train_subset, y_train_subset)
    
    y_scores = clf.predict_proba(X_val_flat)[:, 1]
    y_pred = (y_scores > 0.5).astype(int)
    
    metrics = {
        "roc_auc": float(roc_auc_score(y_val, y_scores)),
        "pr_auc": float(average_precision_score(y_val, y_scores)),
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_val, y_pred)),
        "f1": float(f1_score(y_val, y_pred, zero_division=0))
    }
    
    return metrics


def extract_features(dataset, max_samples=None):
    """Extract mel spectrograms from dataset."""
    print(f"Extracting features from {len(dataset)} samples...")
    
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    
    features = []
    labels = []
    
    for i, batch in enumerate(tqdm(loader)):
        mel = batch["mel"]
        label = batch["label"]
        
        features.append(mel.numpy())
        labels.append(label.numpy())
        
        if max_samples and (i + 1) * 32 >= max_samples:
            break
    
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    if max_samples:
        features = features[:max_samples]
        labels = labels[:max_samples]
    
    return features, labels


def main():
    print("=" * 60)
    print("DIAGNOSTIC BASELINES")
    print("=" * 60)
    
    # Load datasets
    train_ds = MIMIIDataset(cfg, split="train")
    val_ds = MIMIIDataset(cfg, split="val")
    
    print(f"Train dataset: {len(train_ds)} samples")
    print(f"Val dataset: {len(val_ds)} samples")
    
    # Extract features (subset for efficiency)
    max_train_samples = 2000
    max_val_samples = 1000
    
    X_train, y_train = extract_features(train_ds, max_samples=max_train_samples)
    X_val, y_val = extract_features(val_ds, max_samples=max_val_samples)
    
    print(f"Train features: {X_train.shape}")
    print(f"Val features: {X_val.shape}")
    
    # Run baselines
    results = {}
    
    # Majority class baseline
    print("\n" + "=" * 60)
    print("MAJORITY CLASS BASELINE")
    print("=" * 60)
    results["majority_class"] = majority_class_baseline(y_val)
    print(f"ROC-AUC: {results['majority_class']['roc_auc']:.4f}")
    print(f"Accuracy: {results['majority_class']['accuracy']:.4f}")
    print(f"Balanced Accuracy: {results['majority_class']['balanced_accuracy']:.4f}")
    
    # Random score baseline
    print("\n" + "=" * 60)
    print("RANDOM SCORE BASELINE")
    print("=" * 60)
    results["random_score"] = random_score_baseline(y_val)
    print(f"ROC-AUC: {results['random_score']['roc_auc']:.4f}")
    print(f"Accuracy: {results['random_score']['accuracy']:.4f}")
    print(f"Balanced Accuracy: {results['random_score']['balanced_accuracy']:.4f}")
    
    # Logistic regression baseline
    print("\n" + "=" * 60)
    print("LOGISTIC REGRESSION BASELINE")
    print("=" * 60)
    try:
        results["logistic_regression"] = logistic_regression_baseline(X_train, y_train, X_val, y_val)
        print(f"ROC-AUC: {results['logistic_regression']['roc_auc']:.4f}")
        print(f"Accuracy: {results['logistic_regression']['accuracy']:.4f}")
        print(f"Balanced Accuracy: {results['logistic_regression']['balanced_accuracy']:.4f}")
    except Exception as e:
        print(f"Error: {e}")
        results["logistic_regression"] = {"error": str(e)}
    
    # Random forest baseline
    print("\n" + "=" * 60)
    print("RANDOM FOREST BASELINE")
    print("=" * 60)
    try:
        results["random_forest"] = random_forest_baseline(X_train, y_train, X_val, y_val)
        print(f"ROC-AUC: {results['random_forest']['roc_auc']:.4f}")
        print(f"Accuracy: {results['random_forest']['accuracy']:.4f}")
        print(f"Balanced Accuracy: {results['random_forest']['balanced_accuracy']:.4f}")
    except Exception as e:
        print(f"Error: {e}")
        results["random_forest"] = {"error": str(e)}
    
    # Save results
    output_dir = Path("artifacts/baselines")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        else:
            return obj
    
    results_serializable = convert_to_serializable(results)
    
    with open(output_dir / "diagnostic_baselines.json", 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_dir / 'diagnostic_baselines.json'}")
    
    # Comparison with CHAAD
    print("\n" + "=" * 60)
    print("COMPARISON WITH CHAAD")
    print("=" * 60)
    
    chaad_auc = 0.6000  # From evaluation audit
    
    print(f"CHAAD ROC-AUC: {chaad_auc:.4f}")
    print(f"Majority Class: {results['majority_class']['roc_auc']:.4f}")
    print(f"Random Score: {results['random_score']['roc_auc']:.4f}")
    
    if "logistic_regression" in results and "error" not in results["logistic_regression"]:
        print(f"Logistic Regression: {results['logistic_regression']['roc_auc']:.4f}")
    
    if "random_forest" in results and "error" not in results["random_forest"]:
        print(f"Random Forest: {results['random_forest']['roc_auc']:.4f}")
    
    # Classification
    print("\n" + "=" * 60)
    print("FINAL CLASSIFICATION")
    print("=" * 60)
    
    if "logistic_regression" in results and "error" not in results["logistic_regression"]:
        if results["logistic_regression"]["roc_auc"] > chaad_auc + 0.1:
            print("FINAL STATUS: CHAAD-SPECIFIC FAILURE")
            print("Simple baselines outperform CHAAD")
        elif results["logistic_regression"]["roc_auc"] < chaad_auc - 0.1:
            print("FINAL STATUS: CHAAD SHOWS PROMISE")
            print("CHAAD outperforms simple baselines")
        else:
            print("FINAL STATUS: INCONCLUSIVE")
            print("CHAAD and baselines have similar performance")
    else:
        print("FINAL STATUS: INCONCLUSIVE")
        print("Could not train baselines successfully")


if __name__ == "__main__":
    main()
