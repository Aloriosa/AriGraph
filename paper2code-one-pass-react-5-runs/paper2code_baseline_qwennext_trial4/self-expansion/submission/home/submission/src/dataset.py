"""
Dataset loading and preprocessing for SEMA algorithm
"""
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

def load_cifar100_data(batch_size=32):
    """
    Load CIFAR-100 dataset
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader

def load_imagenet_data(batch_size=32):
    """
    Load ImageNet dataset
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.429, 0.422, 0.406])
    ])
    
    train_dataset = datasets.ImageNet(root='./data', split='train', download=False, transform=transform)
    test_dataset = datasets.ImageNet(root='./data', split='val', download=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader