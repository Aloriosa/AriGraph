"""
Full evaluation pipeline.

- Loads a small ImageNet validation split (200 images) from
  data/imagenet_sample/ (already included in the repo).
- Loads a small ImageNet‑Sketch OOD split from data/imagenet_sketch/
- Evaluates two pre‑trained models:
    * torchvision.models.resnet50 (vision‑only)
    * huggingface transformers CLIP ViT‑B‑32 (vision‑language)
- Computes:
    * Top‑1 accuracy on the ID split
    * Average LCA distance on the ID split
    * Top‑1 accuracy on the OOD split
- Correlates ID LCA distance with OOD accuracy across the two models
  (simple linear regression).
- Stores the results in results.json.
"""

import json
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Tuple, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from transformers import CLIPProcessor, CLIPModel

from lca_metric import get_lca_distance

# ---------- Configuration ----------
BATCH_SIZE = 32
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths to the small datasets (included in the repo)
DATA_ROOT = Path(__file__).parent / "data"

ID_DATASET_DIR = DATA_ROOT / "imagenet_sample"
OOD_DATASET_DIR = DATA_ROOT / "imagenet_sketch"

# Models to evaluate
MODEL_CONFIGS = [
    {
        "name": "resnet50",
        "type": "vision",
        "loader": lambda: models.resnet50(pretrained=True).to(DEVICE),
    },
    {
        "name": "clip-vit-base-patch32",
        "type": "clip",
        "loader": lambda: (
            CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32"),
            CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE),
        ),
    },
]

# ---------- Helper functions ----------
def load_imagenet_like_dataset(root: Path):
    """
    Load an ImageNet‑style dataset using ImageFolder.
    Images must be organized as: root/<class_name>/<img>.jpg
    """
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    dataset = datasets.ImageFolder(root, transform=transform)
    return dataset

def predict_vision_model(model, dataloader):
    """Return predictions (class indices) and targets for a vision model."""
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(DEVICE)
            logits = model(images)
            pred = logits.argmax(dim=1).cpu().numpy()
            preds.extend(pred.tolist())
            targets.extend(labels.numpy().tolist())
    return preds, targets

def predict_clip_model(processor, model, dataloader):
    """Return predictions (class indices) and targets for CLIP."""
    # CLIP expects a list of PIL images; we will reload them
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for batch in dataloader:
            # batch: (image, target)
            images, labels = batch
            # CLIP expects PIL images; convert back
            pil_images = [transforms.ToPILImage()(img.cpu()) for img in images]
            inputs = processor(
                text=[f"a photo of a {cls_name}" for cls_name in processor.tokenizer.batch_decode(labels)],
                images=pil_images,
                return_tensors="pt",
                padding=True,
            )
            # We only need image embeddings
            image_embeds = model.get_image_features(**inputs)
            logits = image_embeds @ model.visual_projection.weight.T
            pred = logits.argmax(dim=1).cpu().numpy()
            preds.extend(pred.tolist())
            targets.extend(labels.numpy().tolist())
    return preds, targets

def compute_accuracy(preds, targets) -> float:
    return sum(p == t for p, t in zip(preds, targets)) / len(preds)

def compute_avg_lca(preds, targets) -> float:
    distances = [get_lca_distance(p, t) for p, t in zip(preds, targets)]
    return sum(distances) / len(distances)

def linear_regression(xs, ys):
    """
    Simple linear regression (least squares).
    Returns slope, intercept, r2, pearson r.
    """
    import numpy as np
    xs = np.array(xs)
    ys = np.array(ys)
    n = len(xs)
    if n < 2:
        return 0, 0, 0, 0
    x_mean = xs.mean()
    y_mean = ys.mean()
    slope = ((xs - x_mean) * (ys - y_mean)).sum() / ((xs - x_mean) ** 2).sum()
    intercept = y_mean - slope * x_mean
    pred = slope * xs + intercept
    ss_res = ((ys - pred) ** 2).sum()
    ss_tot = ((ys - y_mean) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    # Pearson correlation
    r = ((xs - x_mean) * (ys - y_mean)).sum() / (
        np.sqrt(((xs - x_mean) ** 2).sum() * ((ys - y_mean) ** 2).sum())
    )
    return slope, intercept, r2, r

# ---------- Main ----------
def main():
    # Load datasets
    print("Loading ID dataset...")
    id_dataset = load_imagenet_like_dataset(ID_DATASET_DIR)
    id_loader = DataLoader(
        id_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )
    print(f"ID dataset size: {len(id_dataset)} images")

    print("Loading OOD dataset...")
    ood_dataset = load_imagenet_like_dataset(OOD_DATASET_DIR)
    ood_loader = DataLoader(
        ood_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )
    print(f"OOD dataset size: {len(ood_dataset)} images")

    results = {}
    id_lca_list = []
    ood_acc_list = []

    for cfg in MODEL_CONFIGS:
        print(f"\n=== Evaluating {cfg['name']} ===")
        if cfg["type"] == "vision":
            model = cfg["loader"]()
            model.to(DEVICE)
            preds, targets = predict_vision_model(model, id_loader)
        else:  # clip
            processor, model = cfg["loader"]()
            model.to(DEVICE)
            # For simplicity we use the same loader for CLIP (image folder)
            # but we need actual class names for CLIP text prompts.
            # Here we use the class indices directly.
            preds, targets = predict_clip_model(processor, model, id_loader)

        # Compute metrics
        acc = compute_accuracy(preds, targets)
        lca = compute_avg_lca(preds, targets)
        print(f"Top‑1 accuracy (ID): {acc*100:.2f}%")
        print(f"Average LCA distance (ID): {lca:.3f}")

        # OOD evaluation
        if cfg["type"] == "vision":
            ood_preds, ood_tgts = predict_vision_model(model, ood_loader)
        else:
            ood_preds, ood_tgts = predict_clip_model(processor, model, ood_loader)
        ood_acc = compute_accuracy(ood_preds, ood_tgts)
        print(f"Top‑1 accuracy (OOD): {ood_acc*100:.2f}%")

        results[cfg["name"]] = {
            "id_accuracy": acc,
            "id_lca": lca,
            "ood_accuracy": ood_acc,
        }

        id_lca_list.append(lca)
        ood_acc_list.append(ood_acc)

    # Correlation analysis across models
    print("\n=== Correlation between ID LCA and OOD accuracy ===")
    slope, intercept, r2, r = linear_regression(id_lca_list, ood_acc_list)
    print(f"Slope: {slope:.3f}, Intercept: {intercept:.3f}")
    print(f"R²: {r2:.3f}, Pearson r: {r:.3f}")

    results["correlation"] = {
        "slope": slope,
        "intercept": intercept,
        "r2": r2,
        "pearson": r,
    }

    # Save results
    out_path = Path("results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {out_path}")

if __name__ == "__main__":
    main()