"""
Download CIFAR‑10 and extract 10 images of class 'cat' as the target domain.
The dataset is used both for training the classifier and for the few‑shot
fine‑tuning of the diffusion model.
"""
import os
import random
from pathlib import Path

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

TARGET_DIR = Path("data/target_images")


def download_and_prepare(num_samples=10, seed=42):
    """Download CIFAR‑10 and extract `num_samples` images of class 3 ('cat')."""
    random.seed(seed)
    os.makedirs(TARGET_DIR, exist_ok=True)
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.CIFAR10(
        root="data", train=False, download=True, transform=transform
    )
    cat_indices = [i for i, (_, label) in enumerate(dataset) if label == 3]  # cat
    selected = random.sample(cat_indices, num_samples)
    subset = Subset(dataset, selected)
    # Save images to disk
    for idx, (img, _) in enumerate(subset):
        img_pil = transforms.ToPILImage()(img)
        img_pil.save(TARGET_DIR / f"cat_{idx}.png")


def load_target_dataset(batch_size=4):
    """Return a DataLoader with the target images."""
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.ImageFolder(root=TARGET_DIR, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)