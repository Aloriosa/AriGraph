#!/usr/bin/env python3
"""
Evaluation script that:
1. Loads a pretrained vision model (ResNet‑18).
2. Downloads the CIFAR‑10 test set.
3. Maps CIFAR‑10 classes to WordNet synsets.
4. Computes top‑1 accuracy and mean LCA distance.
5. Saves per‑sample results to results.csv.
"""

import os
import json
import csv
from pathlib import Path
from collections import defaultdict

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm

from lca.lca_distance import compute_lca_distance
from lca.hierarchy import SynsetMapper

# --------------------------------------------------------------------------- #
# 1. Configuration
# --------------------------------------------------------------------------- #
MODEL_NAME = "resnet18"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
NUM_WORKERS = 4
RESULTS_CSV = "results.csv"

# --------------------------------------------------------------------------- #
# 2. CIFAR‑10 class to WordNet synset ID mapping
# --------------------------------------------------------------------------- #
CIFAR10_CLASS_TO_WNID = {
    0: "n02691156",  # airplane
    1: "n02958343",  # automobile (car)
    2: "n02055957",  # bird
    3: "n02123045",  # cat
    4: "n02130308",  # deer
    5: "n02106662",  # dog
    6: "n01641577",  # frog
    7: "n02374451",  # horse
    8: "n01443537",  # ship
    9: "n02834778",  # truck
}

# --------------------------------------------------------------------------- #
# 3. Load model
# --------------------------------------------------------------------------- #
model = torchvision.models.resnet18(pretrained=True)
model = model.to(DEVICE)
model.eval()

# --------------------------------------------------------------------------- #
# 4. Load CIFAR‑10 test set
# --------------------------------------------------------------------------- #
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

test_set = torchvision.datasets.CIFAR10(
    root=str(Path.home() / ".cache" / "torchvision"),
    train=False,
    download=True,
    transform=transform,
)

test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

# --------------------------------------------------------------------------- #
# 5. Prepare synset mapper
# --------------------------------------------------------------------------- #
synset_mapper = SynsetMapper(CIFAR10_CLASS_TO_WNID)

# --------------------------------------------------------------------------- #
# 6. Evaluation loop
# --------------------------------------------------------------------------- #
results = []
correct = 0
total = 0
lca_distances = []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Evaluating"):
        images = images.to(DEVICE)
        outputs = model(images)
        _, preds = outputs.max(1)

        for pred, true in zip(preds.cpu().numpy(), labels.numpy()):
            total += 1
            is_correct = int(pred == true)
            correct += is_correct

            # LCA distance
            lca_dist = compute_lca_distance(
                true, pred, synset_mapper
            )
            lca_distances.append(lca_dist)

            results.append({
                "image_index": total - 1,
                "true_label": test_set.classes[true],
                "pred_label": test_set.classes[pred],
                "lca_distance": lca_dist,
                "correct": is_correct,
            })

# --------------------------------------------------------------------------- #
# 7. Compute metrics
# --------------------------------------------------------------------------- #
accuracy = correct / total
mean_lca = np.mean(lca_distances)

print(f"\nTop‑1 Accuracy: {accuracy:.4f}")
print(f"Mean LCA Distance: {mean_lca:.4f}")

# --------------------------------------------------------------------------- #
# 8. Save results to CSV
# --------------------------------------------------------------------------- #
with open(RESULTS_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "image_index", "true_label", "pred_label",
        "lca_distance", "correct"
    ])
    writer.writeheader()
    writer.writerows(results)

print(f"\nResults saved to {RESULTS_CSV}")