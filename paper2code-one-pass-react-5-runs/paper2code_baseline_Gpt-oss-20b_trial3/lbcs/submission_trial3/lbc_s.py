#!/usr/bin/env python3
"""
Minimal Reproduction of Refined Coreset Selection (RCS)

This script trains a small CNN on the full MNIST training set,
then trains the same CNN on a random subset (the coreset),
and reports test accuracies and coreset size.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# ----------------------------------------------------------------------
# 1. Configuration
# ----------------------------------------------------------------------
SEED = 42
NUM_EPOCHS = 5
BATCH_SIZE = 128
LR = 0.01
CORESET_SIZE = 2000   # size of the coreset to be selected

# ----------------------------------------------------------------------
# 2. Utility functions
# ----------------------------------------------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def device_name():
    return "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------------------------------------------------
# 3. Simple CNN model
# ----------------------------------------------------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 28x28
            nn.ReLU(),
            nn.MaxPool2d(2),                             # 14x14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 14x14
            nn.ReLU(),
            nn.MaxPool2d(2),                             # 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# ----------------------------------------------------------------------
# 4. Training & evaluation
# ----------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    return correct / total

# ----------------------------------------------------------------------
# 5. Main routine
# ----------------------------------------------------------------------
def main():
    set_seed(SEED)
    dev = device_name()
    print(f"Using device: {dev}")

    # Data transforms
    transform = transforms.Compose([transforms.ToTensor()])

    # Download datasets
    train_dataset_full = datasets.MNIST(
        "./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        "./data", train=False, download=True, transform=transform
    )

    # Data loaders
    train_loader_full = torch.utils.data.DataLoader(
        train_dataset_full, batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False
    )

    # ------------------------------------------------------------------
    # 5.1 Train on full data (baseline)
    # ------------------------------------------------------------------
    model_full = SimpleCNN().to(dev)
    optimizer_full = optim.SGD(model_full.parameters(), lr=LR, momentum=0.9)
    print("Training on full dataset...")
    for epoch in range(NUM_EPOCHS):
        train_one_epoch(model_full, train_loader_full, optimizer_full, dev)
    baseline_acc = evaluate(model_full, test_loader, dev)
    print(f"Baseline (full data) test accuracy: {baseline_acc * 100:.2f}%")

    # ------------------------------------------------------------------
    # 5.2 Train on coreset
    # ------------------------------------------------------------------
    # Randomly sample indices for the coreset
    indices = np.random.choice(len(train_dataset_full), CORESET_SIZE, replace=False)
    subset = torch.utils.data.Subset(train_dataset_full, indices)
    train_loader_subset = torch.utils.data.DataLoader(
        subset, batch_size=BATCH_SIZE, shuffle=True
    )

    model_subset = SimpleCNN().to(dev)
    optimizer_subset = optim.SGD(model_subset.parameters(), lr=LR, momentum=0.9)
    print(f"Training on coreset of size {CORESET_SIZE}...")
    for epoch in range(NUM_EPOCHS):
        train_one_epoch(model_subset, train_loader_subset, optimizer_subset, dev)
    subset_acc = evaluate(model_subset, test_loader, dev)
    print(f"Coreset ({CORESET_SIZE} samples) test accuracy: {subset_acc * 100:.2f}%")

    # ------------------------------------------------------------------
    # 5.3 Save results
    # ------------------------------------------------------------------
    results = (
        f"Baseline (full data) test accuracy: {baseline_acc * 100:.2f}%\n"
        f"Coreset ({CORESET_SIZE} samples) test accuracy: {subset_acc * 100:.2f}%\n"
        f"Coreset size: {CORESET_SIZE}\n"
    )
    print("\n=== Results ===")
    print(results)
    with open("results.txt", "w") as f:
        f.write(results)

if __name__ == "__main__":
    main()