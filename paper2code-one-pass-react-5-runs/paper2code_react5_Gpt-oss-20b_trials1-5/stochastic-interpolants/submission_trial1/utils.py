import random
import math
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import datasets, transforms

# ------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def psnr(img1: torch.Tensor, img2: torch.Tensor, max_val: float = 1.0) -> float:
    mse = F.mse_loss(img1, img2, reduction="mean").item()
    if mse == 0:
        return float("inf")
    return 20 * math.log10(max_val) - 10 * math.log10(mse)

# ------------------------------------------------------------------
# Dataset wrappers
# ------------------------------------------------------------------
class InpaintDataset(Dataset):
    """
    CIFAR‑10 images with a random 4×4 mask applied.
    Returns: corrupted image, mask, target image
    """
    def __init__(self, split: str = "train", transform=None):
        self.cifar = datasets.CIFAR10(root=".", train=split=="train", download=True)
        self.transform = transform

    def __len__(self):
        return len(self.cifar)

    def __getitem__(self, idx):
        img, _ = self.cifar[idx]
        if self.transform:
            img = self.transform(img)  # (3, H, W) in [0,1]
        else:
            img = transforms.ToTensor()(img)

        # Mask
        mask = torch.ones_like(img)
        h, w = img.shape[1], img.shape[2]
        x0 = random.randint(0, w - 4)
        y0 = random.randint(0, h - 4)
        mask[:, y0:y0+4, x0:x0+4] = 0.0

        # Corrupted image
        noise = torch.randn_like(img) * 0.1
        corrupted = mask * img + (1 - mask) * noise
        return corrupted, mask, img

class SuperResDataset(Dataset):
    """
    CIFAR‑10 images downsampled by factor 2 (32→64).
    Returns: base image (upsampled low‑res + noise), low‑res image, target high‑res image
    """
    def __init__(self, split: str = "train", transform=None):
        self.cifar = datasets.CIFAR10(root=".", train=split=="train", download=True)
        self.transform = transform

    def __len__(self):
        return len(self.cifar)

    def __getitem__(self, idx):
        img, _ = self.cifar[idx]
        if self.transform:
            img = self.transform(img)  # (3, H, W) in [0,1]
        else:
            img = transforms.ToTensor()(img)

        # Low‑res version (32→16)
        low = F.interpolate(img.unsqueeze(0), scale_factor=0.5,
                            mode="bilinear", align_corners=False)[0]
        # Upsample back to high‑res for base
        up = F.interpolate(low.unsqueeze(0), scale_factor=2.0,
                           mode="bilinear", align_corners=False)[0]
        # Add small Gaussian noise
        noise = torch.randn_like(up) * 0.1
        base = up + noise
        return base, low, img