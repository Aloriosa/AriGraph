import torch
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms
import numpy as np

class CIFAR100TaskDataset:
    """
    Create class‑incremental splits for CIFAR‑100.
    Each task contains `task_size` classes.
    """
    def __init__(self, task_idx: int, task_size: int = 10, train: bool = True):
        self.task_idx = task_idx
        self.task_size = task_size
        self.train = train
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4) if train else transforms.CenterCrop(32),
            transforms.RandomHorizontalFlip() if train else transforms.RandomHorizontalFlip(0),
            transforms.ToTensor(),
            transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                                 (0.2673342858792401, 0.25643846291708836, 0.27615047132568404)),
        ])
        self.dataset = datasets.CIFAR100(root='./data', train=train, download=True,
                                         transform=transform)

        # Determine class indices for this task
        all_classes = list(range(100))
        np.random.shuffle(all_classes)
        self.task_classes = all_classes[task_idx * task_size : (task_idx + 1) * task_size]

        # Filter indices belonging to this task
        self.indices = [i for i, (_, label) in enumerate(self.dataset) if label in self.task_classes]
        self.subset = Subset(self.dataset, self.indices)

    def get_loader(self, batch_size: int = 128, shuffle: bool = True):
        return DataLoader(self.subset, batch_size=batch_size, shuffle=shuffle,
                          num_workers=4, pin_memory=True)

    def get_task_classes(self):
        """Return class indices of this task."""
        return self.task_classes