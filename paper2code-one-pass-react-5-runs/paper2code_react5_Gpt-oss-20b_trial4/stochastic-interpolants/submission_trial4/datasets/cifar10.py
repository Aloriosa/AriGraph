import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset

class CIFAR10Inpaint(Dataset):
    """CIFAR‑10 with random block masks."""
    def __init__(self, root, train=True, transform=None, mask_prob=0.3, mask_size=8):
        self.dataset = datasets.CIFAR10(root=root, train=train, download=True, transform=transform)
        self.mask_prob = mask_prob
        self.mask_size = mask_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]  # img: Tensor, C×H×W
        # Create mask
        H, W = img.shape[1:]
        mask = torch.ones_like(img)
        # Randomly zero out a block
        if np.random.rand() < self.mask_prob:
            top = np.random.randint(0, H - self.mask_size)
            left = np.random.randint(0, W - self.mask_size)
            mask[:, top:top+self.mask_size, left:left+self.mask_size] = 0.0
        # Corrupted base (noise inside mask)
        noise = torch.randn_like(img)
        x0 = img.clone()
        x0[:, mask.squeeze(0)==0] = noise[:, mask.squeeze(0)==0]
        return x0, img, mask

class CIFAR10SuperRes(Dataset):
    """CIFAR‑10 super‑resolution: low‑res as conditioning."""
    def __init__(self, root, train=True, transform=None, scale=4):
        self.dataset = datasets.CIFAR10(root=root, train=train, download=True, transform=transform)
        self.scale = scale

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]  # C×H×W
        # Down‑sample and up‑sample
        low = torch.nn.functional.interpolate(img.unsqueeze(0), scale_factor=1/self.scale, mode='bicubic', align_corners=False).squeeze(0)
        low_up = torch.nn.functional.interpolate(low.unsqueeze(0), size=img.shape[1:], mode='bicubic', align_corners=False).squeeze(0)
        return low_up, img