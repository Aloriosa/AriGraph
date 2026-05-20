"""
Dataset creation and loading utilities.
Creates a 10‑shot target dataset if it does not exist.
"""

import os
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

TARGET_DIR = "data/target"
NUM_SAMPLES = 10
IMAGE_SIZE = 32  # CIFAR‑10 size

class TargetDataset(Dataset):
    """
    Simple dataset that loads all images from TARGET_DIR.
    If the directory is empty, random noise images are generated.
    """
    def __init__(self, transform=None):
        super().__init__()
        self.transform = transform
        os.makedirs(TARGET_DIR, exist_ok=True)
        files = [p for p in os.listdir(TARGET_DIR) if p.endswith(('.png', '.jpg', '.jpeg'))]
        if len(files) == 0:
            # Create 10 random images
            for i in range(NUM_SAMPLES):
                arr = np.random.randint(0, 256, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
                img = Image.fromarray(arr)
                img.save(os.path.join(TARGET_DIR, f"sample_{i}.png"))
            files = [p for p in os.listdir(TARGET_DIR) if p.endswith(('.png', '.jpg', '.jpeg'))]
        self.files = [os.path.join(TARGET_DIR, f) for f in files]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path).convert("RGB")
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
        if self.transform is not None:
            img = self.transform(img)
        return img

def collate_fn(batch):
    """
    Collate function that stacks images into a tensor.
    """
    return torch.stack(batch, dim=0)