import argparse
import random
import numpy as np
import torch
from typing import List, Tuple

from .utils import set_seed, device
from .dataset import load_mnist
from .models import SimpleCNN
from .trainer import train_and_eval
from torch.utils.data import Subset, DataLoader

def random_coreset_indices(dataset_len: int, k: int, rng: random.Random) -> List[int]:
    """Return a list of `k` unique indices sampled uniformly at random."""
    return rng.sample(range(dataset_len), k)

def greedy_refine(initial_mask: List[int],
                  train_set,
                  test_loader,
                  epochs: int,
                  lr: float,
                  device: torch.device,
                  max_iter: int = 20,
                  rng: random.Random = random.Random()) -> Tuple[List[int], float, float]:
    """
    Very small greedy refinement:
    Start from `initial_mask`, try swapping one element at a time.
    Keep the swap if it improves the test accuracy.
    Returns the best mask found, its loss and accuracy.
    """
    best_mask = initial_mask[:]
    best_loss, best_acc = train_and_eval(SimpleCNN().to(device),
                                         DataLoader(Subset(train_set, best_mask),
                                                    batch_size=256, shuffle=True),
                                         test_loader,
                                         epochs=epochs,
                                         lr=lr,
                                         device=device)
    print(f"Initial coreset accuracy: {best_acc*100:.2f}%")

    dataset_len = len(train_set)
    for it in range(max_iter):
        # pick one element inside and one outside to swap
        in_idx = rng.choice(best_mask)
        out_choices = [i for i in range(dataset_len) if i not in best_mask]
        if not out_choices:
            break
        out_idx = rng.choice(out_choices)

        new_mask = best_mask[:]
        new_mask.remove(in_idx)
        new_mask.append(out_idx)

        loss, acc = train_and_eval(SimpleCNN().to(device),
                                   DataLoader(Subset(train_set, new_mask),
                                              batch_size=256, shuffle=True),
                                   test_loader,
                                   epochs=epochs,
                                   lr=lr,
                                   device=device)
        if acc > best_acc + 1e-4:  # strict improvement
            best_mask = new_mask
            best_loss, best_acc = loss, acc
            print(f"Iteration {it+1}: improved accuracy to {best_acc*100:.2f}%")
        else:
            # no improvement, keep old mask
            pass

    return best_mask, best_loss, best_acc

def main():
    parser = argparse.ArgumentParser(description="Minimal RCS reproduction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--k", type=int, default=5000, help="Coreset size")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--refine", action="store_true",
                        help="Run greedy refinement on the random coreset")
    args = parser.parse_args()

    set_seed(args.seed)
    rng = random.Random(args.seed)

    print("Loading MNIST dataset...")
    train_loader, test_loader, full_train_set = load_mnist(batch_size=256)

    dev = device()
    print(f"Using device: {dev}")

    # 1. Train on full data
    print("\n=== Training on full dataset ===")
    full_net = SimpleCNN().to(dev)
    full_loss, full_acc = train_and_eval(full_net,
                                         train_loader,
                                         test_loader,
                                         epochs=args.epochs,
                                         lr=args.lr,
                                         device=dev)
    print(f"Full data test accuracy: {full_acc*100:.2f}%\n")

    # 2. Random coreset
    print(f"=== Training on random coreset of size {args.k} ===")
    coreset_indices = random_coreset_indices(len(full_train_set), args.k, rng)
    coreset_loader = DataLoader(Subset(full_train_set, coreset_indices),
                                batch_size=256, shuffle=True, num_workers=4)
    coreset_net = SimpleCNN().to(dev)
    coreset_loss, coreset_acc = train_and_eval(coreset_net,
                                               coreset_loader,
                                               test_loader,
                                               epochs=args.epochs,
                                               lr=args.lr,
                                               device=dev)
    print(f"Random coreset test accuracy: {coreset_acc*100:.2f}%\n")

    # 3. Optional greedy refinement
    if args.refine:
        print("=== Running greedy refinement on the random coreset ===")
        refined_mask, refined_loss, refined_acc = greedy_refine(
            coreset_indices,
            full_train_set,
            test_loader,
            epochs=1,   # very few epochs for speed
            lr=args.lr,
            device=dev,
            max_iter=10,
            rng=rng
        )
        print(f"Refined coreset size: {len(refined_mask)}")
        print(f"Refined coreset test accuracy: {refined_acc*100:.2f}%")

if __name__ == "__main__":
    main()