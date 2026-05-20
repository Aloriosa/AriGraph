#!/usr/bin/env python3
"""
Lexicographic Bilevel Coreset Selection (LBCS) – lightweight reproduction

This script implements a simplified but faithful version of the algorithm
described in the paper *Refined Coreset Selection: Towards Minimal
Coreset Size under Model Performance Constraints*.
It focuses on the core ideas:
  * Inner‑loop training of a neural network on a selected coreset.
  * Outer‑loop mask optimisation with a lexicographic objective
    (model performance first, coreset size second).
  * Evaluation of the primary objective on the *full* training set,
    as defined in the paper.
  * A user‑configurable relative compromise ε on the primary objective.

Author: open-source adaptation (2026)
"""

import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm


# --------------------------------------------------------------------------- #
# 1. Utility functions
# --------------------------------------------------------------------------- #
def set_random_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def accuracy(output, target):
    """Compute accuracy for a batch."""
    _, pred = torch.max(output, dim=1)
    return (pred == target).float().mean().item()


# --------------------------------------------------------------------------- #
# 2. Simple CNN model (as used in the paper)
# --------------------------------------------------------------------------- #
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, in_channels=1):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14x14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# --------------------------------------------------------------------------- #
# 3. Training / Evaluation helpers
# --------------------------------------------------------------------------- #
def train_one_epoch(
    model, loader, criterion, optimizer, device, epoch, verbose=True
):
    model.train()
    epoch_loss = 0.0
    epoch_acc = 0.0
    for inputs, targets in tqdm(loader, desc=f"Epoch {epoch}", disable=not verbose):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * inputs.size(0)
        epoch_acc += accuracy(outputs, targets) * inputs.size(0)

    return epoch_loss / len(loader.dataset), epoch_acc / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    loss_total = 0.0
    acc_total = 0.0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss_total += loss.item() * inputs.size(0)
            acc_total += accuracy(outputs, targets) * inputs.size(0)
    return loss_total / len(loader.dataset), acc_total / len(loader.dataset)


# --------------------------------------------------------------------------- #
# 4. Outer loop: Lexicographic optimisation
# --------------------------------------------------------------------------- #
def lexicographic_compare(
    f1_new, f2_new, f1_best, f2_best, epsilon
):
    """
    Return True if (f1_new, f2_new) is lexicographically better than the incumbent
    (f1_best, f2_best).  The primary objective is f1 (lower is better).
    If f1_new is strictly better we accept it.  If it is within
    (1+epsilon)*f1_best we consider the secondary objective f2.
    """
    # Primary objective
    if f1_new < f1_best - 1e-8:
        return True
    # Compromise region
    if f1_new <= f1_best * (1.0 + epsilon) + 1e-8:
        # Secondary objective: smaller coreset size is better (f2 is the size)
        return f2_new < f2_best
    return False


def lbc_solve(
    train_set,
    test_loader,
    k_init,
    epsilon,
    outer_iters,
    inner_epochs,
    batch_size,
    lr,
    device,
    output_dir,
):
    """
    Main LBCS routine.
    """
    # Ensure reproducible mask updates
    set_random_seed(42)

    n_samples = len(train_set)
    indices = np.arange(n_samples)

    # Determine number of classes
    num_classes = getattr(train_set, "num_classes", 10)

    # Initial mask: random k samples
    mask = np.zeros(n_samples, dtype=bool)
    mask[np.random.choice(n_samples, k_init, replace=False)] = True
    best_mask = mask.copy()
    best_f2 = int(mask.sum())

    # Create a loader for the *full* training set (used for f1 evaluation)
    full_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=False, num_workers=2
    )

    # Evaluate incumbent on full training set
    model = SimpleCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    subset = torch.utils.data.Subset(train_set, indices[mask])
    loader = torch.utils.data.DataLoader(
        subset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    for epoch in range(1, inner_epochs + 1):
        train_one_epoch(
            model, loader, criterion, optimizer, device, epoch, verbose=False
        )
    loss, acc = evaluate(model, full_loader, criterion, device)
    best_f1 = loss  # lower loss is better

    # Outer loop
    for t in tqdm(range(outer_iters), desc="Outer loop"):
        # Propose new mask by flipping a few bits
        new_mask = mask.copy()
        # Flip 5% of bits randomly
        num_flips = max(1, int(0.05 * n_samples))
        flip_indices = np.random.choice(
            n_samples, num_flips, replace=False
        )
        new_mask[flip_indices] = ~new_mask[flip_indices]

        # Prevent mask from becoming empty
        if new_mask.sum() == 0:
            new_mask[np.random.choice(n_samples)] = True

        # Train on new mask
        subset = torch.utils.data.Subset(train_set, indices[new_mask])
        loader = torch.utils.data.DataLoader(
            subset, batch_size=batch_size, shuffle=True, num_workers=2
        )
        model = SimpleCNN(num_classes=num_classes).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        for epoch in range(1, inner_epochs + 1):
            train_one_epoch(
                model, loader, criterion, optimizer, device, epoch, verbose=False
            )

        # Evaluate new mask on the full training set
        loss_new, acc_new = evaluate(model, full_loader, criterion, device)
        f1_new = loss_new
        f2_new = int(new_mask.sum())

        # Lexicographic comparison
        if lexicographic_compare(f1_new, f2_new, best_f1, best_f2, epsilon):
            mask = new_mask
            best_mask = new_mask
            best_f1 = f1_new
            best_f2 = f2_new
            best_acc = acc_new

    # Final evaluation on best mask
    final_acc = evaluate_on_best(
        best_mask, train_set, test_loader, device, lr, batch_size, epochs=5
    )

    # Write results
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(output_dir, "results.txt"), "w") as f:
        f.write(f"Final coreset size: {best_f2}\n")
        f.write(f"Final test accuracy: {final_acc * 100:.2f}%\n")
        f.write(f"Final test loss: {best_f1:.4f}\n")
    print(f"Results written to {os.path.join(output_dir, 'results.txt')}")
    return best_f2, final_acc


def evaluate_on_best(
    mask, train_set, test_loader, device, lr, batch_size, epochs=5
):
    """
    Train a fresh model on the best mask and evaluate accuracy.
    """
    subset = torch.utils.data.Subset(train_set, np.where(mask)[0])
    loader = torch.utils.data.DataLoader(
        subset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    num_classes = getattr(train_set, "num_classes", 10)
    model = SimpleCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        train_one_epoch(
            model, loader, criterion, optimizer, device, epoch, verbose=False
        )
    _, acc = evaluate(model, test_loader, criterion, device)
    return acc


# --------------------------------------------------------------------------- #
# 5. Main entry point
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="LBCS simplified implementation")
    parser.add_argument(
        "--dataset",
        type=str,
        default="fmnist",
        help="Dataset to use: fmnist, svhn, cifar10",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=200,
        help="Initial coreset size (number of samples to select)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.2,
        help="Relative compromise for the primary objective",
    )
    parser.add_argument(
        "--outer-iterations",
        type=int,
        default=200,
        help="Number of outer iterations",
    )
    parser.add_argument(
        "--inner-epochs",
        type=int,
        default=3,
        help="Number of epochs to train on the coreset each outer iteration",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for inner training",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate for inner training",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to store results",
    )
    args = parser.parse_args()

    set_random_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    # Load datasets
    if args.dataset == "fmnist":
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )
        train_set = torchvision.datasets.FashionMNIST(
            root="./data", train=True, download=True, transform=transform
        )
        test_set = torchvision.datasets.FashionMNIST(
            root="./data", train=False, download=True, transform=transform
        )
    elif args.dataset == "svhn":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        train_set = torchvision.datasets.SVHN(
            root="./data", split="train", download=True, transform=transform
        )
        test_set = torchvision.datasets.SVHN(
            root="./data", split="test", download=True, transform=transform
        )
    elif args.dataset == "cifar10":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        train_set = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform
        )
        test_set = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=256, shuffle=False, num_workers=2
    )

    # Run LBCS
    lbc_solve(
        train_set,
        test_loader,
        k_init=args.k,
        epsilon=args.epsilon,
        outer_iters=args.outer_iterations,
        inner_epochs=args.inner_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()