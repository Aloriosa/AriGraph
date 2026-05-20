#!/usr/bin/env python
"""
Entry point for training and evaluating SEMA on CIFAR-100 class‑incremental
setting.  The script can be run with `python train_sema.py`.
"""
import argparse
import torch

from src.models.sema import SEMAViT
from src.utils.dataset import get_cifar100_incremental
from src.utils.training import train_sema

def parse_args():
    parser = argparse.ArgumentParser(description="SEMA CIFAR‑100 training")
    parser.add_argument("--tasks", type=int, default=10,
                        help="Number of incremental tasks (default: 10)")
    parser.add_argument("--batch", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--expansions", type=int, default=3,
                        help="Number of layers allowed to expand")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    return parser.parse_args()

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Build dataset loaders
    loaders = get_cifar100_incremental(num_tasks=args.tasks,
                                       batch_size=args.batch,
                                       seed=args.seed)

    # Instantiate model
    model = SEMAViT(expansion_layers=args.expansions).to(model.device)

    # Train over tasks
    train_sema(model, loaders, output_dir="./checkpoints")

if __name__ == "__main__":
    main()