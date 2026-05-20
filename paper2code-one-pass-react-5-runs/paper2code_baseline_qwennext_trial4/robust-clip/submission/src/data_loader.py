import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

def get_data_loader(batch_size=16, num_workers=4, dataset_name='imagenet'):
    """
    Get data loader for ImageNet dataset
    This is a simplified version for reproduction purposes
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.456, 0.456, 0.456])
    ])
    
    if dataset_name == 'imagenet':
        # For reproduction, we'll use a dummy dataset
        # In a real implementation, we would use ImageNet
        # For reproduction, we'll create a dummy dataset of size 1000
        data = torch.randn(1000, 3, 224, 224)
        labels = torch.randint(0, 100, (1000,))
        dataset = torch.utils.data.TensorDataset(data, labels)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return data_loader

def get_test_data_loader(batch_size=16, num_workers=4):
    """
    Get test data loader
    """
    return get_data_loader(batch_size, num_workers)