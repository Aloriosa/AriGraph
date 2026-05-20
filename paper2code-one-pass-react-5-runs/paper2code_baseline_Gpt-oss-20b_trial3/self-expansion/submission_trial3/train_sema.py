#!/usr/bin/env python3
"""
A minimal implementation of the Self‑Expansion of pre‑trained Models with
Mixture of Adapters (SEMA) for class‑incremental learning on CIFAR‑10.

This script:
  1. Loads a pretrained ViT‑B/16 model from timm.
  2. Freezes all weights of the backbone.
  3. Adds a small modular adapter and a representation descriptor (AE)
     to the last transformer block.
  4. Performs self‑expansion: during the first epoch of a new task the
     reconstruction error of the AE is used to decide whether to add a
     new adapter.
  5. Trains the adapters and the AE on each task.
  6. Evaluates accuracy on the test set after each task.
  7. Saves a JSON file with per‑task accuracies.

The code is intentionally lightweight to keep runtime and storage
requirements small.  Results are not meant to match the paper exactly
but demonstrate the overall pipeline.
"""

import json
import os
import math
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import timm

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
NUM_TASKS = 5            # Number of incremental tasks
CLASSES_PER_TASK = 2     # Number of classes per task (CIFAR‑10 → 5 tasks)
BATCH_SIZE = 128
EPOCHS_PER_TASK = 5
LEARNING_RATE = 1e-3
ADAPTER_HIDDEN = 64
AE_HIDDEN = 128
EXPANSION_THRESHOLD = 1.5   # Z‑score threshold for self‑expansion
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def get_cifar10_dataloaders(task_id):
    """
    Returns train and test DataLoader objects for the given task.
    Task i contains classes [2*i, 2*i+1] from CIFAR‑10.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])

    # Full training set
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=transform)
    test_set  = torchvision.datasets.CIFAR10(root='./data', train=False,
                                             download=True, transform=transform)

    # Filter classes for the current task
    selected_classes = list(range(task_id * CLASSES_PER_TASK,
                                  (task_id + 1) * CLASSES_PER_TASK))
    train_indices = [i for i, (_, label) in enumerate(train_set) if label in selected_classes]
    test_indices  = [i for i, (_, label) in enumerate(test_set)  if label in selected_classes]

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(train_set, train_indices),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader  = torch.utils.data.DataLoader(
        torch.utils.data.Subset(test_set, test_indices),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    return train_loader, test_loader, selected_classes

# --------------------------------------------------------------------------- #
# Model components
# --------------------------------------------------------------------------- #
class Adapter(nn.Module):
    """ Lightweight 2‑layer adapter (down‑up) """
    def __init__(self, dim, hidden):
        super().__init__()
        self.down = nn.Linear(dim, hidden, bias=False)
        self.up   = nn.Linear(hidden, dim, bias=False)
        self.act  = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.down(x)).matmul(self.up.weight)

class AEDescriptor(nn.Module):
    """ Auto‑encoder used as representation descriptor """
    def __init__(self, dim, hidden):
        super().__init__()
        self.encoder = nn.Linear(dim, hidden, bias=False)
        self.decoder = nn.Linear(hidden, dim, bias=False)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        z = self.act(self.encoder(x))
        recon = self.decoder(z)
        return recon

class ExpandableRouter(nn.Module):
    """ Soft‑max weighting router over adapters """
    def __init__(self, dim, num_adapters):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim, num_adapters))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        # x: (B, D)
        logits = x @ self.weight   # (B, K)
        return torch.softmax(logits, dim=-1)  # (B, K)

# --------------------------------------------------------------------------- #
# Main SEMA model
# --------------------------------------------------------------------------- #
class SEMA(nn.Module):
    def __init__(self, backbone, dim, num_classes):
        super().__init__()
        self.backbone = backbone   # Frozen ViT
        self.backbone.eval()       # Ensure eval mode
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.dim = dim
        self.num_classes = num_classes

        # We only add adapters to the last transformer block
        # Store lists of adapters, descriptors, and routers
        self.adapters   = nn.ModuleList()
        self.descriptors = nn.ModuleList()
        self.routers   = nn.ModuleList()

        # Final classification head
        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, x):
        # x: (B, 3, H, W)
        features = self.backbone.forward_features(x)  # (B, N, D)
        # Use the [CLS] token representation
        cls = features[:, 0, :]                      # (B, D)

        # Mix adapters
        if len(self.adapters) > 0:
            # Compute router weights
            router = self.routers[0]  # Only one router for last layer
            w = router(cls)          # (B, K)
            # Compute adapter outputs
            adapt_out = torch.zeros_like(cls)
            for k, adapter in enumerate(self.adapters):
                adapt_out += w[:, k: k+1] * adapter(cls)
            cls = cls + adapt_out

        logits = self.classifier(cls)
        return logits

# --------------------------------------------------------------------------- #
# Training utilities
# --------------------------------------------------------------------------- #
def train_one_epoch(model, loader, adapters, descriptors, optimizer,
                    epoch, task_id, criterion, ae_criterion):
    model.train()
    for X, y in tqdm(loader, desc=f"Task {task_id+1} Epoch {epoch+1}", leave=False):
        X = X.to(DEVICE)
        y = y.to(DEVICE)

        # Forward through backbone only
        with torch.no_grad():
            feats = model.backbone.forward_features(X)
        cls = feats[:, 0, :].detach()

        # Forward through adapters and classifier
        logits = model(cls)
        loss_cls = criterion(logits, y)

        # Loss for representation descriptors
        loss_rd = 0.0
        for rd in descriptors:
            recon = rd(cls)
            loss_rd += ae_criterion(recon, cls)

        loss = loss_cls + loss_rd
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in loader:
            X = X.to(DEVICE)
            y = y.to(DEVICE)
            feats = model.backbone.forward_features(X)
            cls = feats[:, 0, :]
            logits = model(cls)
            pred = torch.argmax(logits, dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total

# --------------------------------------------------------------------------- #
# Self‑expansion logic
# --------------------------------------------------------------------------- #
def should_expand(descriptors, cls, threshold):
    """Decide whether to add a new adapter."""
    z_scores = []
    for rd in descriptors:
        with torch.no_grad():
            recon = rd(cls)
            err = torch.mean((cls - recon) ** 2, dim=1)  # (B,)
        mean = err.mean().item()
        std  = err.std().item() + 1e-8
        z = (err - mean) / std
        z_scores.append(z.mean().item())
    # If all z-scores exceed threshold → expand
    return all(z > threshold for z in z_scores)

# --------------------------------------------------------------------------- #
# Main training loop
# --------------------------------------------------------------------------- #
def main():
    # Load pretrained ViT-B/16
    backbone = timm.create_model('vit_base_patch16_224', pretrained=True)
    backbone.eval()
    backbone = backbone.to(DEVICE)

    # Determine feature dimension
    dim = backbone.embed_dim

    # Initialize SEMA model
    model = SEMA(backbone, dim, num_classes=CLASSES_PER_TASK * NUM_TASKS).to(DEVICE)

    # Loss functions
    criterion = nn.CrossEntropyLoss()
    ae_criterion = nn.MSELoss()

    # Optimizer will be updated with new parameters when adapters/routers are added
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=LEARNING_RATE)

    # Store per‑task accuracies
    results = {"per_task_accuracy": []}

    # Iterate over tasks
    for task_id in range(NUM_TASKS):
        print(f"\n===== TASK {task_id+1}/{NUM_TASKS} =====")
        train_loader, test_loader, class_mapping = get_cifar10_dataloaders(task_id)

        # ----- First epoch: decide on expansion -----
        # Forward through backbone only
        with torch.no_grad():
            X_batch, _ = next(iter(train_loader))
            X_batch = X_batch.to(DEVICE)
            feats = backbone.forward_features(X_batch)
            cls = feats[:, 0, :]

        # Check if we need to add a new adapter
        expand = should_expand(model.descriptors, cls, EXPANSION_THRESHOLD)
        if expand:
            print("  → Adding new adapter and descriptor")
            # Append new adapter
            new_adapter = Adapter(dim, ADAPTER_HIDDEN).to(DEVICE)
            model.adapters.append(new_adapter)

            # Append new descriptor
            new_descriptor = AEDescriptor(dim, AE_HIDDEN).to(DEVICE)
            model.descriptors.append(new_descriptor)

            # Update router to accommodate new adapter
            if len(model.routers) == 0:
                router = ExpandableRouter(dim, len(model.adapters)).to(DEVICE)
                model.routers.append(router)
            else:
                # Expand router weight matrix
                old_router = model.routers[0]
                new_weight = nn.Parameter(torch.zeros(dim, len(model.adapters)))
                with torch.no_grad():
                    new_weight[:, :-1] = old_router.weight
                old_router.weight = new_weight

            # Re‑create optimizer with new parameters
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=LEARNING_RATE)

        # ----- Train on this task -----
        for epoch in range(EPOCHS_PER_TASK):
            train_one_epoch(model, train_loader, model.adapters,
                            model.descriptors, optimizer,
                            epoch, task_id, criterion, ae_criterion)

        # ----- Evaluate -----
        acc = evaluate(model, test_loader)
        print(f"  Test accuracy on Task {task_id+1}: {acc:.2f}%")
        results["per_task_accuracy"].append(acc)

    # Save results
    os.makedirs("output", exist_ok=True)
    with open("output/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nTraining finished. Results saved to output/results.json")

if __name__ == "__main__":
    main()