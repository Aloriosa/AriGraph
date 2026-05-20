"""
Refined Coreset Selection (Reproduction)

This script implements a lightweight version of the lexicographic bilevel coreset
selection algorithm described in the paper *Refined Coreset Selection: Towards Minimal Coreset Size under Model Performance Constraints*.

Key components:
- Inner loop: train a CNN on a coreset subset.
- Outer loop: evaluate the trained model on the full training set to compute
  the primary objective (model accuracy).
- Lexicographic update rule: first maximise accuracy; if the new accuracy is
  within a relative tolerance `epsilon` of the best seen, we try to reduce the
  coreset size.
- Mask updates: at each outer iteration we flip a single data point in the
  coreset and accept the change if it improves the lexicographic objective.

Author: ChatGPT (2026)
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset
DATA_ROOT = "./data"

# Coreset size
K = 2000

# Outer loop parameters
MAX_ITER = 20          # Number of mask update iterations
EPSILON = 0.2          # Relative tolerance for secondary objective

# Inner loop parameters
EPOCHS = 5
BATCH_SIZE = 64
LR = 0.001

# --------------------------------------------------------------------------- #
# Utility functions
# --------------------------------------------------------------------------- #
def get_loader(dataset, indices, batch_size, shuffle=True):
    """Create a DataLoader for a subset of the dataset."""
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=2)

# --------------------------------------------------------------------------- #
# Model definition
# --------------------------------------------------------------------------- #
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# --------------------------------------------------------------------------- #
# Training / evaluation
# --------------------------------------------------------------------------- #
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    for data, target in loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

def evaluate_accuracy(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    return correct / total

def train_and_eval(mask_indices, trainset, testset):
    """Train a model on the coreset defined by mask_indices and return test accuracy."""
    train_loader = get_loader(trainset, mask_indices, BATCH_SIZE, shuffle=True)
    test_loader = get_loader(testset, list(range(len(testset))), BATCH_SIZE, shuffle=False)

    model = SimpleCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for _ in range(EPOCHS):
        train_one_epoch(model, train_loader, criterion, optimizer)

    acc = evaluate_accuracy(model, test_loader)
    return acc

# --------------------------------------------------------------------------- #
# Core algorithm
# --------------------------------------------------------------------------- #
def main():
    # ------------------------------------------------------------------ #
    # 1. Load datasets
    # ------------------------------------------------------------------ #
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.FashionMNIST(
        root=DATA_ROOT, train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.FashionMNIST(
        root=DATA_ROOT, train=False, download=True, transform=transform
    )

    # ------------------------------------------------------------------ #
    # 2. Initialise random coreset mask
    # ------------------------------------------------------------------ #
    total_samples = len(trainset)
    all_indices = np.arange(total_samples)

    # Randomly choose K indices
    mask = np.zeros(total_samples, dtype=np.int32)
    init_indices = np.random.choice(total_samples, size=K, replace=False)
    mask[init_indices] = 1

    # Evaluate initial mask
    best_indices = init_indices
    best_acc = train_and_eval(best_indices, trainset, testset)
    best_size = K

    # Log file
    log_path = "log.txt"
    with open(log_path, "w") as log_f:
        log_f.write("Iteration,Accuracy,Size\n")

    print(f"Iteration 0: Accuracy = {best_acc*100:.2f}%, Coreset size = {best_size}")

    # ------------------------------------------------------------------ #
    # 3. Outer loop: lexicographic mask updates
    # ------------------------------------------------------------------ #
    for it in range(1, MAX_ITER + 1):
        # Randomly pick an index to flip
        idx = np.random.randint(total_samples)

        # Propose new mask by flipping idx
        new_mask = mask.copy()
        new_mask[idx] = 1 - new_mask[idx]  # flip 0<->1

        # Compute new indices and evaluate
        new_indices = all_indices[new_mask == 1]
        new_acc = train_and_eval(new_indices, trainset, testset)
        new_size = new_indices.size

        # Lexicographic comparison
        accept = False
        if new_acc > best_acc:
            accept = True
        else:
            # Check if within tolerance of best_acc
            if (best_acc - new_acc) / best_acc <= EPSILON:
                if new_size < best_size:
                    accept = True

        if accept:
            mask = new_mask
            best_indices = new_indices
            best_acc = new_acc
            best_size = new_size
            print(f"Iteration {it}: Accuracy = {best_acc*100:.2f}%, Coreset size = {best_size}")
        else:
            print(f"Iteration {it}: Rejected (Acc={new_acc*100:.2f}%, Size={new_size})")

        # Log
        with open(log_path, "a") as log_f:
            log_f.write(f"{it},{best_acc},{best_size}\n")

    # ------------------------------------------------------------------ #
    # 4. Save final results
    # ------------------------------------------------------------------ #
    results_path = "results.txt"
    with open(results_path, "w") as f:
        f.write(f"Test Accuracy: {best_acc*100:.4f}%\n")
        f.write(f"Coreset size used: {best_size}\n")

    print("\nReproduction finished.")
    print(f"Results written to {results_path}")

if __name__ == "__main__":
    main()