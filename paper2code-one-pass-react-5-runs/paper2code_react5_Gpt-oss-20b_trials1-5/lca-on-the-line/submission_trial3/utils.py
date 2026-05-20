# utils.py – helper functions for data loading and metrics
# -------------------------------------------------------

import os
import json
import math
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    r2_score,
    pearsonr,
    spearmanr,
    kendalltau,
)

# --------------------------------------------------
# 1. Dataset helpers
# --------------------------------------------------
def get_imagenet_val(split_dir="imagenet_val"):
    """
    Load the ImageNet validation set (1K classes).
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    return ImageFolder(root=split_dir, transform=transform)

def get_ood_dataset(name, split_dir="ood_datasets"):
    """
    Load an ImageNet‑based OOD dataset using the HuggingFace datasets library.
    The dataset is downloaded the first time.
    """
    import datasets
    dataset = datasets.load_dataset(name, split="validation")
    # Convert to torchvision format
    imgs = [transform_image(x["image"]) for x in tqdm(dataset, desc=f"Loading {name}")]
    labels = [x["label"] for x in dataset]
    return torch.utils.data.TensorDataset(torch.stack(imgs), torch.tensor(labels))

def transform_image(img):
    """Apply the same transform as ImageNet."""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    return transform(img)

# --------------------------------------------------
# 2. Metric helpers
# --------------------------------------------------
def top1_accuracy(preds, labels):
    """Compute top‑1 accuracy."""
    return (preds == labels).float().mean().item()

def top1_accuracy_from_logits(logits, labels):
    preds = logits.argmax(dim=1)
    return top1_accuracy(preds, labels)

def compute_lca_scores(model, dataloader, device, lca_fn, use_soft=False):
    """
    Compute ID LCA and ELCA for a model.
    lca_fn: function that takes (pred_id, gt_id) -> distance
    use_soft: if True, compute ELCA using softmax probabilities
    """
    model.eval()
    lca_total = 0.0
    elca_total = 0.0
    count = 0
    with torch.no_grad():
        for imgs, labels in tqdm(dataloader, desc="LCA eval"):
            imgs = imgs.to(device)
            logits = model(imgs)
            if use_soft:
                probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            for i in range(len(labels)):
                gt_id = f"{labels[i].item():08d}"
                pred_id = f"{preds[i].item():08d}"
                lca_total += lca_fn(pred_id, gt_id)
                if use_soft:
                    elca_total += lca_fn(pred_id, gt_id) * probs[i, preds[i]].item()
            count += len(labels)
    avg_lca = lca_total / count
    avg_elca = elca_total / count
    return avg_lca, avg_elca

# --------------------------------------------------
# 3. Correlation helpers
# --------------------------------------------------
def compute_correlations(df, x_col, y_col):
    """
    Compute R^2, Pearson, Spearman, Kendall for a pair of columns.
    """
    x = df[x_col].values
    y = df[y_col].values
    r2 = r2_score(x, y)
    pear = pearsonr(x, y)[0]
    spearman = spearmanr(x, y)[0]
    kendall = kendalltau(x, y)[0]
    return {
        "R2": r2,
        "Pearson": pear,
        "Spearman": spearman,
        "Kendall": kendall,
    }