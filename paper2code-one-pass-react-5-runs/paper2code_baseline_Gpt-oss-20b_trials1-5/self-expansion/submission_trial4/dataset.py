import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import random
import numpy as np

class IncrementalCIFAR10(Dataset):
    """
    Wraps CIFAR10 to create class‑incremental tasks.
    Each task contains `n_classes_per_task` disjoint classes.
    """
    def __init__(self, root, train=True, transform=None, n_tasks=5, n_classes_per_task=2, seed=42):
        self.n_tasks = n_tasks
        self.n_classes_per_task = n_classes_per_task
        self.seed = seed
        self.transform = transform

        dataset = datasets.CIFAR10(root=root, train=train, download=True)
        data, targets = dataset.data, np.array(dataset.targets)

        # Shuffle classes
        rng = np.random.RandomState(seed)
        all_classes = np.arange(10)
        rng.shuffle(all_classes)

        # Assign classes to tasks
        self.task_classes = []
        for i in range(n_tasks):
            cls = all_classes[i * n_classes_per_task : (i + 1) * n_classes_per_task]
            self.task_classes.append(cls)

        # Build indices per task
        self.task_indices = []
        for cls in self.task_classes:
            idx = np.where(targets == cls)[0]
            self.task_indices.append(idx)

        # Flatten all data for convenience
        self.data = data
        self.targets = targets

    def get_task(self, task_id):
        """
        Returns a DataLoader for the specified task.
        """
        idx = self.task_indices[task_id]
        subset = Subset(self, idx)
        return DataLoader(subset, batch_size=32, shuffle=True, num_workers=2)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        if self.transform:
            img = self.transform(img)
        return img, target