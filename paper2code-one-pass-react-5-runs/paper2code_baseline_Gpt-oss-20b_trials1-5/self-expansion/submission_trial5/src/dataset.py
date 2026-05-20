"""
Utilities for creating incremental CIFAR‑100 splits.
"""
import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
import numpy as np

def get_cifar100_splits(num_tasks: int = 10, batch_size: int = 128):
    """
    Return a list of (train_loader, test_loader) tuples for each incremental task.
    Each task contains an equal number of classes (10 classes per task for CIFAR‑100).
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
    ])

    train_full = datasets.CIFAR100(root="data", train=True,
                                   download=True, transform=transform_train)
    test_full  = datasets.CIFAR100(root="data", train=False,
                                   download=True, transform=transform_test)

    # Map each class to its indices
    class_to_idx = {c: [] for c in range(100)}
    for idx, (_, label) in enumerate(train_full):
        class_to_idx[label].append(idx)

    # Prepare tasks
    tasks = []
    classes = list(range(100))
    np.random.shuffle(classes)
    for t in range(num_tasks):
        task_classes = classes[t*10:(t+1)*10]
        train_indices = []
        test_indices  = []
        for c in task_classes:
            train_indices.extend(class_to_idx[c])
            test_indices.extend([i for i, (_, l) in enumerate(test_full) if l == c])

        train_subset = Subset(train_full, train_indices)
        test_subset  = Subset(test_full, test_indices)

        train_loader = DataLoader(train_subset, batch_size=batch_size,
                                 shuffle=True, num_workers=2, pin_memory=True)
        test_loader  = DataLoader(test_subset,  batch_size=batch_size,
                                 shuffle=False, num_workers=2, pin_memory=True)
        tasks.append((train_loader, test_loader))

    return tasks