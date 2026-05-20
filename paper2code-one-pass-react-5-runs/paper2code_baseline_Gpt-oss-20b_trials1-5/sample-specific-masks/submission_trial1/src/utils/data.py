import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
import os

def get_dataloader(dataset_name, split, batch_size, seed=42, num_workers=4):
    """
    Returns a DataLoader for the requested dataset split.
    Supported datasets:
        cifar10, cifar100, svhn, gtsrb, flowers102, dtd, ucf101,
        food101, eurosat, oxfordpets, sun397
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    if dataset_name == 'cifar10':
        dataset = datasets.CIFAR10(root='data', train=(split=='train'),
                                   download=True, transform=transform)
    elif dataset_name == 'cifar100':
        dataset = datasets.CIFAR100(root='data', train=(split=='train'),
                                    download=True, transform=transform)
    elif dataset_name == 'svhn':
        dataset = datasets.SVHN(root='data', split=split, download=True,
                               transform=transform)
    elif dataset_name == 'gtsrb':
        # A lightweight GTSRB wrapper
        from src.utils.gtsrb import GTSRB
        dataset = GTSRB(root='data', train=(split=='train'))
    elif dataset_name == 'flowers102':
        dataset = datasets.Flowers102(root='data', split=split,
                                      download=True, transform=transform)
    elif dataset_name == 'dtd':
        dataset = datasets.DTD(root='data', split=split,
                               download=True, transform=transform)
    elif dataset_name == 'ucf101':
        dataset = datasets.UCF101(root='data', split=split,
                                 download=True, transform=transform)
    elif dataset_name == 'food101':
        dataset = datasets.Food101(root='data', split=split,
                                   download=True, transform=transform)
    elif dataset_name == 'eurosat':
        dataset = datasets.EuroSAT(root='data', split=split,
                                  download=True, transform=transform)
    elif dataset_name == 'oxfordpets':
        dataset = datasets.OxfordPets(root='data', split=split,
                                      download=True, transform=transform)
    elif dataset_name == 'sun397':
        dataset = datasets.SUN397(root='data', split=split,
                                  download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset {dataset_name}")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=(split=='train'),
                        num_workers=num_workers, pin_memory=True)
    return loader