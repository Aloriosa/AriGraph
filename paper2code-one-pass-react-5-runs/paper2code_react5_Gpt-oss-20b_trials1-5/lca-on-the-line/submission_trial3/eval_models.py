#!/usr/bin/env python3
# eval_models.py – Main driver for model evaluation
# -----------------------------------------------

import os
import argparse
import json
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

from config import (
    MODEL_REGISTRY,
    IMAGENET_VAL_DIR,
    OOD_DATASETS,
    BATCH_SIZE,
    NUM_WORKERS,
    DEVICE,
)
from utils import get_imagenet_val, get_ood_dataset
from lca import lca_distance, elca_probability_distribution
from utils import compute_lca_scores, compute_correlations

# ------------------------------------------------------------------
# 1. Argument parsing
# ------------------------------------------------------------------
parser = argparse.ArgumentParser(description="LCA‑on‑the‑Line evaluation")
parser.add_argument("--download-only", action="store_true",
                    help="Only download datasets (no evaluation)")
parser.add_argument("--run-eval", action="store_true",
                    help="Run the full evaluation pipeline")
args = parser.parse_args()

# ------------------------------------------------------------------
# 2. Dataset preparation
# ------------------------------------------------------------------
def ensure_imagenet():
    if not os.path.isdir(IMAGENET_VAL_DIR):
        raise RuntimeError(f"ImageNet validation directory '{IMAGENET_VAL_DIR}' not found. "
                           "Download it manually to the repository root.")
    print(f"Using ImageNet validation set at {IMAGENET_VAL_DIR}")

def download_ood_datasets():
    for name in OOD_DATASETS:
        print(f"Downloading OOD dataset {name} ...")
        _ = get_ood_dataset(name)  # dataset is cached

# ------------------------------------------------------------------
# 3. Model utilities
# ------------------------------------------------------------------
def load_model(entry):
    if entry["zero_shot"]:
        model, _ = entry["module"]()
    else:
        model = entry["module"]()
    model = model.to(DEVICE)
    return model

# ------------------------------------------------------------------
# 4. Evaluation loop
# ------------------------------------------------------------------
def evaluate():
    ensure_imagenet()

    # Load dataloaders
    id_dataset = get_imagenet_val(IMAGENET_VAL_DIR)
    id_loader = torch.utils.data.DataLoader(
        id_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    # Prepare OOD loaders
    ood_loaders = {}
    for name in OOD_DATASETS:
        ds = get_ood_dataset(name)
        ood_loaders[name] = torch.utils.data.DataLoader(
            ds, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=True
        )

    # Results containers
    id_acc = {}
    ood_acc = {k: {} for k in OOD_DATASETS}
    lca_scores = {}
    elca_scores = {}

    # Iterate over models
    for entry in MODEL_REGISTRY:
        name = entry["name"]
        print(f"\n=== Evaluating {name} ===")
        model = load_model(entry)

        # ID accuracy
        logits = []
        labels = []
        with torch.no_grad():
            for imgs, labs in tqdm(id_loader, desc="ID inference"):
                imgs = imgs.to(DEVICE)
                out = model(imgs)
                logits.append(out.cpu())
                labels.append(labs)
        logits = torch.cat(logits)
        labels = torch.cat(labels)
        top1 = torch.softmax(logits, dim=1).argmax(dim=1).eq(labels).float().mean().item()
        id_acc[name] = top1
        print(f"ID Top‑1 accuracy: {top1:.4f}")

        # LCA / ELCA
        avg_lca, avg_elca = compute_lca_scores(
            model, id_loader, DEVICE, lca_distance, use_soft=True
        )
        lca_scores[name] = avg_lca
        elca_scores[name] = avg_elca
        print(f"Avg LCA: {avg_lca:.4f}, Avg ELCA: {avg_elca:.4f}")

        # OOD accuracy
        for od_name, od_loader in ood_loaders.items():
            logits = []
            labs = []
            with torch.no_grad():
                for imgs, l in tqdm(od_loader, desc=f"{od_name} inference"):
                    imgs = imgs.to(DEVICE)
                    out = model(imgs)
                    logits.append(out.cpu())
                    labs.append(l)
            logits = torch.cat(logits)
            labs = torch.cat(labs)
            top1 = torch.softmax(logits, dim=1).argmax(dim=1).eq(labs).float().mean().item()
            ood_acc[od_name][name] = top1
            print(f"OOD {od_name} Top‑1: {top1:.4f}")

    # Save results
    os.makedirs("results/plots", exist_ok=True)
    pd.DataFrame.from_dict(id_acc, orient="index", columns=["Top1"]).to_csv(
        "results/id_accuracies.csv"
    )
    for od_name in OOD_DATASETS:
        pd.DataFrame.from_dict(ood_acc[od_name], orient="index", columns=["Top1"]).to_csv(
            f"results/ood_{od_name}.csv"
        )
    pd.DataFrame.from_dict(lca_scores, orient="index", columns=["LCA"]).to_csv(
        "results/lca_scores.csv"
    )
    pd.DataFrame.from_dict(elca_scores, orient="index", columns=["ELCA"]).to_csv(
        "results/elca_scores.csv"
    )

    # ------------------------------------------------------------------
    # 5. Correlation analysis
    # ------------------------------------------------------------------
    # Merge ID LCA with each OOD accuracy
    corr_rows = []
    for od_name in OOD_DATASETS:
        df = pd.DataFrame.from_dict(
            {
                "LCA": lca_scores,
                f"{od_name}_Top1": ood_acc[od_name],
            },
            orient="index",
        ).reset_index().rename(columns={"index": "Model"})
        stats = compute_correlations(df, "LCA", f"{od_name}_Top1")
        stats["Dataset"] = od_name
        corr_rows.append(stats)
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv("results/correlation_results.csv", index=False)
    print("\nCorrelation results:")
    print(corr_df)

    # Simple plot for one dataset (ImageNet‑A) – optional
    try:
        import matplotlib.pyplot as plt
        df = pd.DataFrame.from_dict(
            {
                "LCA": lca_scores,
                "ImageNet_A_Top1": ood_acc["imagenet_a"],
            },
            orient="index",
        ).reset_index().rename(columns={"index": "Model"})
        plt.figure(figsize=(6, 4))
        plt.scatter(df["LCA"], df["ImageNet_A_Top1"])
        plt.xlabel("ID LCA distance")
        plt.ylabel("ImageNet‑A Top‑1")
        plt.title("LCA‑on‑the‑Line (ImageNet‑A)")
        plt.tight_layout()
        plt.savefig("results/plots/correlation.png")
        print("\nSaved correlation plot to results/plots/correlation.png")
    except Exception as e:
        print(f"Plotting failed: {e}")

if __name__ == "__main__":
    if args.download_only:
        download_ood_datasets()
        print("Datasets downloaded.")
    if args.run_eval:
        evaluate()