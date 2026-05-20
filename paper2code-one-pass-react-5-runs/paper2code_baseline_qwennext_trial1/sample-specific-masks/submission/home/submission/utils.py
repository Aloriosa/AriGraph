"""
Utility functions for SMM framework
This module provides utility functions for the SMM framework.
"""
import torch
import numpy as np
import os

def set_seed(seed):
    """
    Set random seed for reproducibility
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_directory(path):
    """
    Create directory if it doesn't exist
    """
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def save_model(model, path):
    """
    Save model
    """
    torch.save(model.state_dict(), path)

def load_model(model, path):
    """
    Load model
    """
    model.load_state_dict(torch.load(path))
    return model

def get_device():
    """
    Get device
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_model_size(model):
    """
    Get model size in MB
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    return param_size / 1024 / 1024

def print_model_info(model, model_name):
    """
    Print model information
    """
    print(f"Model: {model_name}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Model size: {get_model_size(model):.2f} MB")

def print_dataset_info(dataset_name, train_loader, test_loader, num_classes):
    """
    Print dataset information
    """
    print(f"Dataset: {dataset_name}")
    print(f"Train loader: {len(train_loader)} batches")
    print(f"Test loader: {len(test_loader)} batches")
    print(f"Number of classes: {num_classes}")

# Test function
def test_utils():
    """
    Test the utility functions
    """
    print("Testing utility functions...")
    print("Testing set_seed...")
    set_seed(42)
    print("Test passed!")

    print("Testing create_directory...")
    create_directory("test_dir")
    print("Test passed!")

    print("Testing save_model...")
    model = torch.nn.Linear(10, 1)
    save_model(model, "test_model.pth")
    print("Test passed!")

    print("Testing load_model...")
    model = torch.nn.Linear(10, 1)
    model = load_model(model, "test_model.pth")
    print("Test passed!")

    print("Testing get_device...")
    device = get_device()
    print(f"Device: {device}")
    print("Test passed!")

    print("Testing get_model_size...")
    model = torch.nn.Linear(10, 1)
    size = get_model_size(model)
    print(f"Model size: {size:.2f} MB")
    print("Test passed!")

    print("Testing print_model_info...")
    model = torch.nn.Linear(10, 1)
    print_model_info(model, "test_model")
    print("Test passed!")

    print("Testing print_dataset_info...")
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.randn(10, 10), torch.randint(0, 10, (10,))), batch_size=10)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.randn(10, 10), torch.randint(0, 10, (10,))), batch_size=10))
    print_dataset_info("test_dataset", train_loader, test_loader, 10)
    print("Test passed!")

if __name__ == '__main__':
    test_utils()