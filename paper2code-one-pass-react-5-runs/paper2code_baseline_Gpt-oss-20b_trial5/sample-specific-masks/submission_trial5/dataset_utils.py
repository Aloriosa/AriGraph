import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import os

# Mapping from dataset name to torchvision dataset class
DATASET_MAP = {
    "cifar10": datasets.CIFAR10,
    "cifar100": datasets.CIFAR100,
    "svhn": datasets.SVHN,
    "gtsrb": datasets.GTSRB,  # Note: GTSRB is not in torchvision; we use a custom loader
}

def get_dataset(name, split="train"):
    """
    Returns a DataLoader for the specified dataset and split.
    """
    if name not in DATASET_MAP:
        raise ValueError(f"Unsupported dataset {name}")

    if name == "gtsrb":
        # Custom loader for GTSRB
        return get_gtsrb(split)
    else:
        root = "./data"
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
        dataset = DATASET_MAP[name](root=root, train=(split=="train"),
                                   download=True, transform=transform)
        loader = data.DataLoader(dataset, batch_size=128,
                                 shuffle=(split=="train"),
                                 num_workers=4, pin_memory=True)
        return loader

def get_gtsrb(split="train"):
    """
    Minimal loader for GTSRB dataset (train/test split).
    The dataset is expected to be in the current directory under
    ./data/GTSRB/Final_Training/Images for training and
    ./data/GTSRB/Final_Test/Images for testing.
    """
    # The official GTSRB dataset contains separate train/test folders.
    # For simplicity, we use the same transforms as for other datasets.
    if split == "train":
        root = "./data/GTSRB/Final_Training/Images"
    else:
        root = "./data/GTSRB/Final_Test/Images"

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    dataset = datasets.ImageFolder(root=root, transform=transform)
    loader = data.DataLoader(dataset, batch_size=128,
                             shuffle=(split=="train"),
                             num_workers=4, pin_memory=True)
    return loader