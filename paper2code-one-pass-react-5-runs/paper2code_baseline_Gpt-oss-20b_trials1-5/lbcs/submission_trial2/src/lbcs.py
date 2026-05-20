#!/usr/bin/env python
# src/lbcs.py
"""
Refined Coreset Selection (LBCS) – simplified implementation.
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

from .simple_cnn import SimpleCNN
from .utils import evaluate, train_one_epoch, create_subset_loader

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def random_mask(n_samples: int, k: int) -> List[int]:
    """Return a random list of k unique indices from 0..n_samples-1."""
    return random.sample(range(n_samples), k)


def mutate_mask(
    current: List[int],
    all_indices: List[int],
    n_swap: int = 5,
) -> List[int]:
    """Swap n_swap elements between current and all_indices."""
    current_set = set(current)
    pool = list(set(all_indices) - current_set)
    if not pool:
        return current  # nothing to swap
    to_remove = random.sample(current, min(n_swap, len(current)))
    to_add = random.sample(pool, min(n_swap, len(pool)))
    new_set = set(current_set) - set(to_remove) | set(to_add)
    return list(new_set)


def lexicographic_better(
    acc_new: float,
    acc_best: float,
    size_new: int,
    size_best: int,
    epsilon: float,
) -> bool:
    """
    Return True if new mask is better under lexicographic order:
        1. Accuracy >= best_accuracy * (1 - epsilon)
        2. If accuracy equal (within 1e-6), size smaller.
    """
    if acc_new > acc_best * (1 - epsilon) + 1e-6:
        return True
    if abs(acc_new - acc_best * (1 - epsilon)) <= 1e-6 and size_new < size_best:
        return True
    return False


# --------------------------------------------------------------------------- #
# Main algorithm
# --------------------------------------------------------------------------- #
def run_lbcs(
    dataset_name: str,
    k: int,
    epsilon: float,
    T: int,
    epochs: int,
    final_epochs: int,
) -> Tuple[int, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything()

    # Load dataset
    transform = transforms.Compose([transforms.ToTensor()])
    if dataset_name.lower() == "mnist":
        full_train = datasets.MNIST(root="data", train=True, download=True, transform=transform)
        full_test = datasets.MNIST(root="data", train=False, download=True, transform=transform)
    elif dataset_name.lower() == "fashion_mnist":
        full_train = datasets.FashionMNIST(root="data", train=True, download=True, transform=transform)
        full_test = datasets.FashionMNIST(root="data", train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    n_samples = len(full_train)
    all_indices = list(range(n_samples))

    # Initial mask
    mask = random_mask(n_samples, k)
    best_mask = mask
    best_size = len(mask)
    # Evaluate baseline accuracy
    loader = create_subset_loader(full_train, mask, batch_size=256, shuffle=False)
    model = SimpleCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for _ in range(epochs):
        train_one_epoch(model, loader, optimizer, device)
    best_accuracy = evaluate(model, DataLoader(full_test, batch_size=256), device)

    print(f"[Init] size={best_size}, acc={best_accuracy:.2f}%")

    # Outer loop
    for t in range(1, T + 1):
        new_mask = mutate_mask(best_mask, all_indices, n_swap=5)

        loader = create_subset_loader(full_train, new_mask, batch_size=256, shuffle=False)
        model = SimpleCNN().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        for _ in range(epochs):
            train_one_epoch(model, loader, optimizer, device)
        acc_new = evaluate(model, DataLoader(full_test, batch_size=256), device)

        new_size = len(new_mask)
        if lexicographic_better(acc_new, best_accuracy, new_size, best_size, epsilon):
            best_accuracy = acc_new
            best_size = new_size
            best_mask = new_mask
            print(
                f"[{t:03d}] improved -> size={best_size}, acc={best_accuracy:.2f}%"
            )
        else:
            # keep old best
            pass

    # Final training on best coreset
    loader = create_subset_loader(full_train, best_mask, batch_size=256, shuffle=True)
    model = SimpleCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for _ in range(final_epochs):
        train_one_epoch(model, loader, optimizer, device)
    final_acc = evaluate(model, DataLoader(full_test, batch_size=256), device)

    print(f"[Final] size={best_size}, final_acc={final_acc:.2f}%")
    return best_size, final_acc


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LBCS – Refined Coreset Selection")
    p.add_argument("--dataset", type=str, default="mnist")
    p.add_argument("--k", type=int, default=2000)
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--T", type=int, default=200)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--final_epochs", type=int, default=10)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    size, acc = run_lbcs(
        args.dataset,
        args.k,
        args.epsilon,
        args.T,
        args.epochs,
        args.final_epochs,
    )
    out = {"coreset_size": size, "test_accuracy": acc}
    Path("results.json").write_text(json.dumps(out, indent=4))
    print(f"Results written to results.json: {out}")


if __name__ == "__main__":
    main()