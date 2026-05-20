import os
import torch
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def get_dataloader(dataset_path: str, batch_size: int, num_workers: int):
    """
    Loads ImageNet (or ImageNet‑style) dataset from folder structure.
    Images are resized to 256x256 and normalized to [-1,1].
    """
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.5]*3, std=[0.5]*3)  # [-1, 1]
    ])
    dataset = ImageFolder(root=dataset_path, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, pin_memory=True)