"""
Utility functions for dataset loading, splitting into tasks,
and evaluation metrics.
"""

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader, Subset

def get_cifar10_tasks(num_tasks=5, classes_per_task=2, batch_size=128, seed=42):
    """
    Load CIFAR‑10 and split into `num_tasks` tasks,
    each containing `classes_per_task` distinct classes.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=False, transform=transform)
    test_set  = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=False, transform=transform)

    # Group indices by class
    class_to_indices = {i: [] for i in range(10)}
    for idx, (_, label) in enumerate(train_set):
        class_to_indices[label].append(idx)

    # Shuffle class order
    all_classes = list(range(10))
    np.random.shuffle(all_classes)
    task_classes = [all_classes[i:i+classes_per_task]
                    for i in range(0, len(all_classes), classes_per_task)]

    tasks = []
    for cls in task_classes:
        indices = []
        for c in cls:
            indices.extend(class_to_indices[c])
        # Shuffle within task
        np.random.shuffle(indices)
        tasks.append(indices)

    # Create DataLoaders per task
    task_loaders = []
    for indices in tasks:
        subset = Subset(train_set, indices)
        loader = DataLoader(subset, batch_size=batch_size,
                            shuffle=True, num_workers=2, drop_last=True)
        task_loaders.append(loader)

    # Test loader (all test data)
    test_loader = DataLoader(test_set, batch_size=batch_size,
                             shuffle=False, num_workers=2)

    return task_loaders, test_loader

def accuracy(preds, targets):
    return (preds == targets).float().mean().item()

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total