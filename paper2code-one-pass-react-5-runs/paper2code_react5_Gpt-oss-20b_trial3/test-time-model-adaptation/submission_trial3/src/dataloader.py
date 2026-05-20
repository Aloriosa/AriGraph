import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from .corruptions import random_corruption

class CIFAR10CDataLoader:
    """CIFAR‑10 test set with optional corruption."""
    def __init__(self, batch_size=64, corruption=False, level=1):
        self.batch_size = batch_size
        self.corruption = corruption
        self.level = level

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

        self.test_set = datasets.CIFAR10(
            root="./data",
            train=False,
            download=True,
            transform=transform,
        )

        self.dataloader = DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

    def __iter__(self):
        for imgs, labels in self.dataloader:
            if self.corruption:
                # apply corruption on the raw pixel values before normalization
                imgs = imgs * 255.0  # bring to [0,255]
                imgs = torch.stack([random_corruption(img, self.level) for img in imgs]) / 255.0
            yield imgs, labels

    def __len__(self):
        return len(self.dataloader)