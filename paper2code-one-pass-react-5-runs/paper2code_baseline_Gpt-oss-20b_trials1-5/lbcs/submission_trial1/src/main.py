import os
import torch
import torchvision
import torchvision.transforms as transforms

from .coreset_selection import lexicographic_bilevel
from .utils import set_seed

def main():
    # Configuration
    k = 200                # predefined coreset size
    epsilon = 0.1          # acceptable performance compromise
    max_iter = 20          # number of outer search iterations
    swap_rate = 0.05       # fraction of samples swapped per iteration
    seed = 42

    set_seed(seed)

    print("=== LBCS on MNIST ===")
    print(f"Predefined coreset size (k): {k}")
    print(f"Performance compromise ε: {epsilon}")
    print(f"Total search iterations: {max_iter}\n")

    # Data loading
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=256, shuffle=False
    )

    # Run LBCS
    best_mask, best_size, best_loss, best_acc = lexicographic_bilevel(
        train_dataset,
        test_loader,
        k=k,
        epsilon=epsilon,
        max_iter=max_iter,
        swap_rate=swap_rate,
        seed=seed,
    )

    print("\n=== Results ===")
    print(f"Best mask size: {best_size}")
    print(f"Test accuracy: {best_acc:.2f}%")
    print(f"Test loss: {best_loss:.4f}")

if __name__ == "__main__":
    main()