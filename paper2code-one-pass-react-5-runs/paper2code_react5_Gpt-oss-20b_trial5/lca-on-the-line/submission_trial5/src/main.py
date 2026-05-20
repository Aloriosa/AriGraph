"""
Main orchestration script.
Loads a small set of models, runs inference on ImageNet‑val,
computes ID accuracy, ID LCA distance, and OOD accuracy (ImageNet‑R).
Finally, prints correlation results and saves a CSV with all metrics.
"""

import os
import torch
import pandas as pd
from src.datasets import get_imagenet_val_loader
from src.models import (load_resnet18, load_resnet50,
                        load_clip_rn50, load_clip_vitb32)
from src.evaluator import evaluate_vm, evaluate_clip, compute_correlations
from pathlib import Path

# --------------------------------------------------------------------------- #
# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --------------------------------------------------------------------------- #
# Load datasets
val_loader = get_imagenet_val_loader(batch_size=64, num_workers=8)

# --------------------------------------------------------------------------- #
# Models to evaluate
models = {
    "ResNet18": load_resnet18(device),
    "ResNet50": load_resnet50(device),
    "CLIP_RN50": load_clip_rn50(device),
    "CLIP_ViT-B32": load_clip_vitb32(device),
}

# --------------------------------------------------------------------------- #
# Dictionaries to store metrics
id_acc = {}
id_lca = {}
ood_acc = {}  # placeholder; would be computed on an OOD dataset

# --------------------------------------------------------------------------- #
# Evaluate each model on ImageNet‑val (ID)
print("\n=== Evaluating on ImageNet‑val (ID) ===")
for name, model in models.items():
    print(f"\nModel: {name}")
    if name.startswith("CLIP"):
        # CLIP models return a tuple (model, preprocess)
        clip_model, preprocess = model
        acc, lca = evaluate_clip(clip_model, preprocess, val_loader,
                                 device, compute_lca=True)
    else:
        acc, lca = evaluate_vm(model, val_loader, device, compute_lca=True)
    id_acc[name] = acc
    id_lca[name] = lca
    print(f"  Top‑1 Accuracy: {acc:.4f}")
    print(f"  Mean LCA distance: {lca:.4f}")

# --------------------------------------------------------------------------- #
# Placeholder for OOD evaluation
# (In a full reproduction, you would load ImageNet‑R, ImageNet‑A, etc.,
#  run the same inference pipeline and fill ood_acc.)
print("\n=== OOD evaluation is not implemented in this minimal example. ===")

# --------------------------------------------------------------------------- #
# Correlation analysis (ID LCA vs. OOD accuracy)
# Here we simply print the correlation between ID LCA and ID accuracy,
# because OOD metrics are missing.
r2, pear = compute_correlations(id_lca, id_acc)
print("\nCorrelation between ID LCA and ID Accuracy:")
print(f"  R²      : {r2:.4f}")
print(f"  Pearson : {pear:.4f}")

# --------------------------------------------------------------------------- #
# Save results to CSV
results = pd.DataFrame({
    "Model": list(id_acc.keys()),
    "ID_Acc": list(id_acc.values()),
    "ID_LCA": list(id_lca.values()),
    "OOD_Acc": [None] * len(id_acc)  # placeholder
})
out_path = Path("results") / "metrics.csv"
out_path.parent.mkdir(exist_ok=True)
results.to_csv(out_path, index=False)
print(f"\nResults written to {out_path}")