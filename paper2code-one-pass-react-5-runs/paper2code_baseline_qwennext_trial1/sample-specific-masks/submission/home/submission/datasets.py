"""
Dataset utilities for SMM framework
This module provides utilities for loading and preprocessing datasets.
"""
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def load_dataset(dataset_name, batch_size=256, image_size=224):
    """
    Load a dataset from torchvision
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.457, 0.407), (0.225, 0.224, 0.221))
    ])
    
    if dataset_name == "cifar10":
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        num_classes = 10
    elif dataset_name == "cifar100":
        train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFIFAR100(root='./data', train=False, download=True, transform=transform)
        num_classes = 100
    elif dataset_name == "svhn":
        train_dataset = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform)
        test_dataset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform)
        num_classes = 10
    elif dataset_name == "gtsrb":
        train_dataset = torchvision.datasets.GTSRB(root='./data', split='train', download=True, transform=transform)
        test_dataset = torchvision.datasets.GTSRB(root='./data', split='test', download=True, transform=transform)
        num_classes = 43
    elif dataset_name == "flowers102":
        train_dataset = torchvision.datasets.Flowers102(root='./data', split='train', download=True, transform=transform)
        test_dataset = torchvision.datasets.Flowers102(root='./data', split='test', download=True, transform=transform)
        num_classes = 102
    elif dataset_name == "dtd":
        train_dataset = torchvision.datasets.DTD(root='./data', split='train', download=True, transform=transform)
        test_dataset = torchvision.datasets.DTD(root='./data', split='test', download=True, transform=transform)
        num_classes = 47
    elif dataset_name == "ucf101":
        train_dataset = torchvision.datasets.UCF101(root='./data', split='train', download=True, transform=transform)
        test_dataset = torchvision.datasets.UCF101(root='./data', split='test', download=True, transform=transform)
        num_classes = 101
    elif dataset_name == "food101":
        train_dataset = torchvision.datasets.Food101(root='./data', split='train', download=True, transform=transform)
        test_dataset = torchvision.datasets.Food101(root='./data', split='test', download=True, transform=transform)
        num_classes = 101
    elif dataset_name == "sun397":
        train_dataset = torchvision.datasets.SUN397(root='./data', split='train', download=True, transform=transform)
        test_dataset = torchvision.datasets.SUN397(root='./data', split='test', download=True, transform=transform)
        num_classes = 397
    elif dataset_name == "eurosat":
        train_dataset = torchvision.datasets.EuroSAT(root='./data', split='train', download=True, transform=transform)
        test_dataset = torchvision.datasets.EuroSAT(root='./data', split='test', download=True, transform=transform)
        num_classes = 10
    elif dataset_name == "oxfordpets":
        train_dataset = torchvision.datasets.OxfordIIITPet(root='./data', split='train', download=True, transform=transform)
        test_dataset = torchvision.datasets.OxfordIIITPet(root='./data', split='test', download=True, transform=transform)
        num_classes = 37
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, num_classes

# Test function
def test_datasets():
    """
    Test the dataset loading utilities
    """
    print("Testing datasets utilities...")
    datasets = ["cifar10", "cifar100", "svhn", "gtsrb", "flowers102", "dtd", "ucf101", "food101", "sun397", "eurosat", "oxfordpets"]
    
    for dataset in datasets:
        print(f"Loading dataset: {dataset}")
        train_loader, test_loader, num_classes = load_dataset(dataset)
        print(f"Train loader: {len(train_loader)} batches")
        print(f"Test loader: {len(test_loader)} batches")
        print(f"Number of classes: {num_classes}")
    
    print("Test passed!")

if __name__ == '__main__':
    test_datasets()