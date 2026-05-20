#!/usr/bin/env python3
"""
Minimal reproduction of the LCA‑on‑the‑Line paper.

The script:
1. Loads a small subset of ImageNet synset IDs via NLTK WordNet.
2. Generates synthetic predictions for a handful of pre‑trained models
   (ResNet‑50, EfficientNet‑B0, CLIP‑RN50).  No heavy model loading is
   performed – this keeps runtimes short while still exercising the
   pipeline.
3. Computes:
   - Top‑1 accuracy on an artificial “ID” set.
   - Average Lowest Common Ancestor (LCA) distance on the same set.
   - Top‑1 accuracy and LCA distance on a synthetic “OOD” set.
4. Calculates the Pearson correlation between ID‑LCA and OOD‑Top‑1.
5. Saves a CSV with per‑model results and a short text summary.

The numbers are synthetic; the goal is to demonstrate that the
pipeline can be executed and that the expected artefacts are produced.
"""

import os
import random
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import pearsonr
import nltk
from nltk.corpus import wordnet as wn

# ------------------------------------------------------------
# Utility functions for LCA computation
# ------------------------------------------------------------

def load_synset_list(n_classes: int = 1000):
    """
    Returns a list of NLTK wordnet synsets (noun sense) used as
    surrogate for the ImageNet 1000‑class hierarchy.
    """
    # Ensure the wordnet corpus is downloaded
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

    # Pick the first `n_classes` noun synsets
    synsets = list(wn.all_synsets('n'))[:n_classes]
    # Keep only the synset ID string (e.g., 'n02124075')
    synset_ids = [s.name() for s in synsets]
    return synset_ids

def get_hypernym_path(synset):
    """
    Return the first hypernym path from the synset to the root.
    """
    # Some synsets have multiple hypernym paths; we take the shortest
    paths = synset.hypernym_paths()
    if not paths:
        return [synset]
    # Choose the path with fewest steps to the root
    return min(paths, key=len)

def lca_distance(syn1, syn2):
    """
    Compute the Lowest Common Ancestor distance between two synsets.
    Distance = depth(syn1) + depth(syn2) - 2 * depth(lca)
    """
    p1 = get_hypernym_path(syn1)
    p2 = get_hypernym_path(syn2)

    # Convert to sets of synset names for quick intersection
    set1 = set(s.name() for s in p1)
    set2 = set(s.name() for s in p2)

    common = set1 & set2
    if not common:
        # If no common ancestor (should not happen), return max depth
        return max(len(p1), len(p2))

    # Find the common synset with greatest depth (i.e., furthest from root)
    lca_name = max(common, key=lambda name: len(p1) - p1.index(wn.synset(name)))
    lca_syn = wn.synset(lca_name)

    depth1 = len(p1) - 1  # depth from syn1 to root
    depth2 = len(p2) - 1
    lca_depth = len(get_hypernym_path(lca_syn)) - 1

    return (depth1 - lca_depth) + (depth2 - lca_depth)

# ------------------------------------------------------------
# Synthetic dataset and evaluation
# ------------------------------------------------------------

def generate_synthetic_batch(num_samples: int, num_classes: int):
    """
    Generate synthetic labels and predictions.
    """
    gt = np.random.randint(0, num_classes, size=num_samples)
    # Random logits: standard normal
    logits = np.random.randn(num_samples, num_classes)
    preds = np.argmax(logits, axis=1)
    # Top‑1 accuracy
    acc = np.mean(preds == gt)
    return gt, preds, acc

def compute_average_lca(gt, preds, synsets):
    """
    Compute mean LCA distance for a batch of predictions.
    """
    distances = [lca_distance(wn.synset(synsets[p]), wn.synset(synsets[g]))
                 for g, p in zip(gt, preds)]
    return np.mean(distances)

# ------------------------------------------------------------
# Main evaluation loop
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LCA‑on‑the‑Line synthetic evaluation")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Directory to store outputs")
    parser.add_argument("--num-samples", type=int, default=5000,
                        help="Number of synthetic samples per set")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Load synthetic WordNet hierarchy
    # --------------------------------------------------------
    synset_ids = load_synset_list(n_classes=1000)
    num_classes = len(synset_ids)

    # --------------------------------------------------------
    # Define model list (we do not actually load the heavy models)
    # --------------------------------------------------------
    models = [
        "resnet50",
        "efficientnet_b0",
        "clip_RN50"
    ]

    # Store results
    rows = []

    for model_name in tqdm(models, desc="Processing models"):
        # 1. ID set
        gt_id, pred_id, acc_id = generate_synthetic_batch(args.num_samples, num_classes)
        lca_id = compute_average_lca(gt_id, pred_id, synset_ids)

        # 2. OOD set (synthetic shift: we simply shuffle the predictions)
        gt_ood, pred_ood, acc_ood = generate_synthetic_batch(args.num_samples, num_classes)
        lca_ood = compute_average_lca(gt_ood, pred_ood, synset_ids)

        rows.append({
            "model": model_name,
            "id_top1_acc": acc_id,
            "id_lca": lca_id,
            "ood_top1_acc": acc_ood,
            "ood_lca": lca_ood
        })

    df = pd.DataFrame(rows)
    csv_path = out_dir / "results.csv"
    df.to_csv(csv_path, index=False)

    # --------------------------------------------------------
    # Correlation analysis
    # --------------------------------------------------------
    corr, _ = pearsonr(df["id_lca"], df["ood_top1_acc"])
    summary = {
        "id_lca_ood_top1_correlation": corr,
        "num_models": len(models)
    }
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Also write a human‑readable text
    txt_path = out_dir / "summary.txt"
    with open(txt_path, "w") as f:
        f.write("Synthetic LCA‑on‑the‑Line Evaluation\n")
        f.write("=================================\n\n")
        f.write(f"Number of models evaluated: {len(models)}\n")
        f.write(f"Pearson correlation between ID LCA and OOD Top‑1 accuracy: {corr:.4f}\n\n")
        f.write("Per‑model results (CSV):\n")
        f.write(f"{csv_path.resolve()}\n")

    print(f"Results written to {out_dir.resolve()}")

if __name__ == "__main__":
    main()