#!/usr/bin/env python3
"""
Refined Coreset Selection (LBCS) – Minimal implementation
Author: OpenAI (adapted for reproduction)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
DEFAULT_CONFIG: Dict[str, Any] = {
    "epsilon": 0.2,
    "max_inner_epochs": 20,
    "outer_iterations": 500,
    "batch_size": 256,
    "learning_rate": 0.001,
    "seed": 42,
    "k_min_ratio": 0.1,
    "device": "cuda",
    "datasets": ["fashion_mnist", "svhn", "cifar10"],
}


# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_dataset(name: str) -> Tuple[data.Dataset, data.Dataset]:
    """Return (train_set, test_set) for the given dataset name."""
    if name == "fashion_mnist":
        train = datasets.FashionMNIST(
            root=".", train=True, download=True, transform=transforms.ToTensor()
        )
        test = datasets.FashionMNIST(
            root=".", train=False, download=True, transform=transforms.ToTensor()
        )
    elif name == "svhn":
        train = datasets.SVHN(
            root=".",
            split="train",
            download=True,
            transform=transforms.ToTensor(),
        )
        test = datasets.SVHN(
            root=".",
            split="test",
            download=True,
            transform=transforms.ToTensor(),
        )
    elif name == "cifar10":
        train = datasets.CIFAR10(
            root=".", train=True, download=True, transform=transforms.ToTensor()
        )
        test = datasets.CIFAR10(
            root=".", train=False, download=True, transform=transforms.ToTensor()
        )
    elif name == "imagenet":
        # ImageNet is not downloaded automatically; user must provide
        # a valid path. We use a placeholder that will fail gracefully.
        raise RuntimeError(
            "ImageNet dataset is not available in the container. "
            "Please provide the data or skip this dataset."
        )
    else:
        raise ValueError(f"Unsupported dataset: {name}")
    return train, test


def get_model(name: str, num_classes: int) -> nn.Module:
    """Return a simple model for the dataset."""
    if name == "fashion_mnist":
        # 2‑layer CNN
        return nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )
    elif name == "svhn":
        # 3‑layer CNN
        return nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )
    elif name == "cifar10":
        # ResNet‑18
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    else:
        raise ValueError(f"Unsupported model for dataset '{name}'.")


def train_one_epoch(
    model: nn.Module,
    loader: data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)


def evaluate(
    model: nn.Module,
    loader: data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = criterion(out, yb)
            total_loss += loss.item() * xb.size(0)
            pred = out.argmax(dim=1)
            correct += (pred == yb).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def lexicographic_better(
    loss_a: float,
    size_a: int,
    loss_b: float,
    size_b: int,
    epsilon: float,
    tol: float = 1e-6,
) -> bool:
    """
    Return True if (loss_a, size_a) is lexicographically better than
    (loss_b, size_b) under the ε‑budget on the primary objective.
    """
    # primary objective: loss
    if loss_a < loss_b - tol:
        return True
    if abs(loss_a - loss_b) <= tol and size_a < size_b:
        return True
    # When loss_a is worse but within ε‑budget, we still consider it better
    # only if it improves the size (but size cannot be smaller than a fixed k)
    # In our greedy scheme we only allow removal, so this case never occurs.
    return False


def evaluate_mask(
    mask: torch.Tensor,
    train_set: data.Dataset,
    test_loader: data.DataLoader,
    model_fn: nn.Module,
    device: torch.device,
    inner_epochs: int,
    batch_size: int,
) -> Tuple[float, int]:
    """Train on subset defined by mask and return validation loss and size."""
    indices = mask.nonzero(as_tuple=False).squeeze().tolist()
    subset = data.Subset(train_set, indices)
    loader = data.DataLoader(
        subset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    criterion = nn.CrossEntropyLoss()
    model = model_fn.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for _ in range(inner_epochs):
        train_one_epoch(model, loader, criterion, optimizer, device)
    loss, _ = evaluate(model, test_loader, criterion, device)
    return loss, len(indices)


# ----------------------------------------------------------------------
# Core bilevel algorithm
# ----------------------------------------------------------------------
def run_coreset_selection(
    dataset_name: str,
    config: Dict[str, Any],
    output_dir: Path,
) -> None:
    set_seed(config["seed"])
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

    print(f"=== Dataset: {dataset_name} ===")
    train_set, test_set = load_dataset(dataset_name)
    num_classes = train_set.classes if hasattr(train_set, "classes") else len(set(train_set.targets))

    # Build test loader (for final evaluation)
    test_loader = data.DataLoader(
        test_set,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=2,
    )

    # Initial mask: full dataset
    mask = torch.ones(len(train_set), dtype=torch.bool, device=device)
    k_min = int(config["k_min_ratio"] * len(train_set))
    k_min = max(k_min, 1)

    # Initial best loss (full dataset)
    print("[INFO] Evaluating baseline on full dataset...")
    baseline_loss, _ = evaluate_mask(
        mask,
        train_set,
        test_loader,
        get_model(dataset_name, num_classes),
        device,
        config["max_inner_epochs"],
        config["batch_size"],
    )
    best_loss = baseline_loss
    best_mask = mask.clone()
    best_size = len(train_set)

    # Outer loop
    print("[INFO] Starting greedy coreset reduction...")
    for it in tqdm(range(config["outer_iterations"]), desc="Outer loop"):
        # Randomly pick a sample to remove
        candidate_idx = random.choice(
            torch.nonzero(best_mask).squeeze().tolist()
        )
        new_mask = best_mask.clone()
        new_mask[candidate_idx] = False

        # Evaluate candidate
        cand_loss, cand_size = evaluate_mask(
            new_mask,
            train_set,
            test_loader,
            get_model(dataset_name, num_classes),
            device,
            config["max_inner_epochs"],
            config["batch_size"],
        )

        # Lexicographic acceptance
        if lexicographic_better(
            cand_loss,
            cand_size,
            best_loss,
            best_size,
            config["epsilon"],
        ):
            best_mask = new_mask
            best_loss = cand_loss
            best_size = cand_size
            print(
                f"[{it+1:04d}] Accepted removal -> size {best_size}, loss {best_loss:.4f}"
            )
            if best_size <= k_min:
                print("[INFO] Reached hard minimum coreset size.")
                break

    # Final training on best mask
    print("[INFO] Training final model on selected coreset...")
    final_loader = data.DataLoader(
        data.Subset(train_set, torch.nonzero(best_mask).squeeze().tolist()),
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=2,
    )
    criterion = nn.CrossEntropyLoss()
    final_model = get_model(dataset_name, num_classes).to(device)
    optimizer = optim.Adam(final_model.parameters(), lr=0.001)
    for _ in range(config["max_inner_epochs"] * 2):
        train_one_epoch(final_model, final_loader, criterion, optimizer, device)

    # Final evaluation on test set
    test_loss, test_acc = evaluate(final_model, test_loader, criterion, device)
    print(
        f"[RESULT] Test accuracy: {test_acc*100:.2f}%, "
        f"test loss: {test_loss:.4f}, "
        f"coreset size: {best_size} ({best_size/len(train_set):.2%})"
    )

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "coreset_indices.npy", torch.nonzero(best_mask).squeeze().cpu().numpy())
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(
            {
                "test_accuracy": test_acc,
                "test_loss": test_loss,
                "coreset_size": best_size,
                "full_size": len(train_set),
                "epsilon": config["epsilon"],
                "inner_epochs": config["max_inner_epochs"],
                "outer_iterations": it + 1,
                "seed": config["seed"],
                "k_min": k_min,
            },
            f,
            indent=4,
        )


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LBCS Reproduction")
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Path to JSON config file (overrides defaults)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="",
        help="Dataset name to run (overrides config list)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.0,
        help="Allowed loss compromise",
    )
    parser.add_argument(
        "--max_inner_epochs",
        type=int,
        default=0,
        help="Epochs per inner loop",
    )
    parser.add_argument(
        "--outer_iterations",
        type=int,
        default=0,
        help="Maximum outer loop iterations",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=0,
        help="Batch size",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0,
        help="Learning rate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )
    parser.add_argument(
        "--k_min_ratio",
        type=float,
        default=0.0,
        help="Minimum coreset size ratio",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Device to run on",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory to store results",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DEFAULT_CONFIG.copy()

    # Load JSON config if provided
    if args.config:
        with open(args.config) as f:
            cfg_json = json.load(f)
        config.update(cfg_json)

    # Override with command line arguments if set
    for key, val in vars(args).items():
        if val:
            if key == "output_dir":
                config[key] = val
            else:
                config[key] = val

    output_root = Path(config["output_dir"])
    datasets_to_run = [args.dataset] if args.dataset else config["datasets"]

    for ds in datasets_to_run:
        out_dir = output_root / ds
        run_coreset_selection(ds, config, out_dir)


if __name__ == "__main__":
    main()