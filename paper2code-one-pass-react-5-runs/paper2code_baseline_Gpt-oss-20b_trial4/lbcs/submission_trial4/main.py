#!/usr/bin/env python3
"""
Refined Coreset Selection Demo
Author: OpenAI Assistant
"""

import os
import random
import argparse
import csv
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

# ----------------------------------
# Utility functions
# ----------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------
# Simple CNN model
# ----------------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)   # 28x28 -> 28x28
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)  # 28x28 -> 28x28
        self.pool = nn.MaxPool2d(2, 2)                 # 28x28 -> 14x14
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# ----------------------------------
# Training & evaluation helpers
# ----------------------------------
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        data, target = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.size(0)
    return total_loss / len(loader.dataset)

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch in loader:
            data, target = batch[0].to(device), batch[1].to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    return correct / len(loader.dataset)

# ----------------------------------
# Core set selection (loss-based)
# ----------------------------------
def select_coreset(dataset, subset_size, device, epochs=3, lr=0.01):
    """
    Train a temporary model, compute loss per sample, select lowest-loss samples.
    """
    loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=2)
    model = SimpleCNN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # Train briefly
    for _ in range(epochs):
        train_one_epoch(model, loader, optimizer, device)

    # Compute loss per sample
    model.eval()
    losses = []
    with torch.no_grad():
        for idx, (data, target) in enumerate(dataset):
            data = data.unsqueeze(0).to(device)
            target = target.unsqueeze(0).to(device)
            output = model(data)
            loss = F.cross_entropy(output, target, reduction='sum')
            losses.append((idx, loss.item()))

    # Sort by loss (ascending) and pick subset_size smallest
    losses.sort(key=lambda x: x[1])
    selected_indices = [idx for idx, _ in losses[:subset_size]]
    return Subset(dataset, selected_indices)

# ----------------------------------
# Main pipeline
# ----------------------------------
def main():
    parser = argparse.ArgumentParser(description="RCS Demo")
    parser.add_argument("--subset-size", type=int, default=200,
                        help="Number of samples in the coreset")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # 1. Load MNIST
    # ------------------------------------------------------------------
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = torchvision.datasets.MNIST(root="./data", train=True,
                                               download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root="./data", train=False,
                                              download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True,
                              num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False,
                             num_workers=2)

    # ------------------------------------------------------------------
    # 2. Train on full training set
    # ------------------------------------------------------------------
    print("\n=== Training on full training set ===")
    full_model = SimpleCNN().to(device)
    optimizer = optim.SGD(full_model.parameters(), lr=0.01, momentum=0.9)
    for epoch in range(5):
        loss = train_one_epoch(full_model, train_loader, optimizer, device)
        acc = evaluate(full_model, test_loader, device)
        print(f"Epoch {epoch+1:02d}: loss={loss:.4f} acc={acc*100:.2f}%")

    full_acc = evaluate(full_model, test_loader, device)
    print(f"\nFull training accuracy: {full_acc*100:.2f}%")

    # ------------------------------------------------------------------
    # 3. Coreset selection
    # ------------------------------------------------------------------
    print("\n=== Selecting coreset ===")
    subset_loader = DataLoader(
        select_coreset(train_dataset, args.subset_size, device),
        batch_size=256, shuffle=True, num_workers=2)

    # ------------------------------------------------------------------
    # 4. Train on coreset
    # ------------------------------------------------------------------
    print("\n=== Training on coreset ===")
    coreset_model = SimpleCNN().to(device)
    optimizer = optim.SGD(coreset_model.parameters(), lr=0.01, momentum=0.9)
    for epoch in range(5):
        loss = train_one_epoch(coreset_model, subset_loader, optimizer, device)
        acc = evaluate(coreset_model, test_loader, device)
        print(f"Epoch {epoch+1:02d}: loss={loss:.4f} acc={acc*100:.2f}%")

    coreset_acc = evaluate(coreset_model, test_loader, device)
    print(f"\nCoreset training accuracy: {coreset_acc*100:.2f}%")

    # ------------------------------------------------------------------
    # 5. Save results
    # ------------------------------------------------------------------
    os.makedirs("outputs", exist_ok=True)
    results_path = os.path.join("outputs", "results.csv")
    with open(results_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "full_accuracy", "subset_accuracy",
                         "subset_size"])
        writer.writerow(["MNIST",
                         f"{full_acc*100:.2f}",
                         f"{coreset_acc*100:.2f}",
                         f"{args.subset_size}"])
    print(f"\nResults written to {results_path}")

if __name__ == "__main__":
    main()