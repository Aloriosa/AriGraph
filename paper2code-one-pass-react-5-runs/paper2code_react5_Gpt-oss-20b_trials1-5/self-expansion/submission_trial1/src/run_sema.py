import os
import csv
import random
import numpy as np
import torch
import torch.utils.data
from torchvision import datasets, transforms
from tqdm import tqdm

from src.sema import create_task_splits, evaluate
from src.model import SEMA


def main():
    # Basic settings
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data transforms (resize to 224 for ViT)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    # CIFAR‑100 dataset
    train_dataset = datasets.CIFAR100(root="data", train=True, transform=transform, download=True)
    test_dataset = datasets.CIFAR100(root="data", train=False, transform=transform, download=True)

    num_tasks = 10
    task_splits = create_task_splits(train_dataset, num_tasks=num_tasks, seed=seed)

    # DataLoader for the whole test set (used for final evaluation)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=256, shuffle=False, num_workers=4
    )

    # Initialize model
    model = SEMA(num_classes=100, device=device).to(device)

    # Results logging
    results = []

    # Train task by task
    for t_idx, (train_idx, test_idx) in enumerate(task_splits, 1):
        print(f"\n=== Training Task {t_idx} ===")
        # Add a new adapter for this task
        model.add_adapter()

        # Create DataLoader for this task
        train_sampler = SubsetRandomSampler(train_idx)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=64, sampler=train_sampler, num_workers=4
        )

        # Train on the current task
        model.train_on_task(
            train_loader,
            epochs=3,          # few epochs for quick demo
            lr_cls=1e-3,
            lr_adapt=1e-3,
            lr_rd=1e-3,
        )

        # Evaluate on all seen classes so far
        # Build a test loader that contains only seen classes
        seen_classes = np.unique(train_dataset.targets[:train_idx.max() + 1])
        test_subset = torch.utils.data.Subset(
            test_dataset,
            np.where(np.isin(test_dataset.targets, seen_classes))[0]
        )
        test_loader_seen = torch.utils.data.DataLoader(
            test_subset, batch_size=256, shuffle=False, num_workers=4
        )
        acc = evaluate(model, test_loader_seen, device)
        print(f"Task {t_idx} accuracy on seen classes: {acc:.4f}")
        results.append((t_idx, acc))

    # Final average accuracy
    avg_acc = sum(acc for _, acc in results) / len(results)
    print(f"\nFinal average accuracy over all tasks: {avg_acc:.4f}")

    # Write results to CSV
    os.makedirs("output", exist_ok=True)
    csv_path = os.path.join("output", "results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task", "accuracy"])
        for t, acc in results:
            writer.writerow([t, f"{acc:.4f}"])
        writer.writerow(["average", f"{avg_acc:.4f}"])

    print(f"\nResults written to {csv_path}")


if __name__ == "__main__":
    main()