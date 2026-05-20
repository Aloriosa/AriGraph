import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import os

def get_cifar100_tasks(num_tasks=10, seed=42):
    """
    Split CIFAR‑100 into `num_tasks` tasks with non‑overlapping 10 classes each.
    """
    dataset = datasets.CIFAR100(root="./data", train=True, download=True,
                                transform=transforms.Compose([
                                    transforms.Resize(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])
                                ]))
    targets = np.array(dataset.targets)
    classes = np.arange(100)
    rng = np.random.default_rng(seed)
    rng.shuffle(classes)
    task_classes = [classes[i*10:(i+1)*10] for i in range(num_tasks)]
    task_indices = []
    for cls in task_classes:
        idx = np.where(np.isin(targets, cls))[0]
        task_indices.append(idx)
    return dataset, task_indices

def get_test_loader(dataset, batch_size=128):
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

def get_task_loader(dataset, idxs, batch_size=128):
    subset = Subset(dataset, idxs)
    return DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=4)