import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

def get_dataset(name, train=True, download=True):
    if name.lower() == "mnist":
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = datasets.MNIST(root="./data", train=train, download=download, transform=transform)
    elif name.lower() == "cifar10":
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = datasets.CIFAR10(root="./data", train=train, download=download, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {name}")
    return dataset

def create_mask(n_samples, mask_indices):
    mask = torch.zeros(n_samples, dtype=torch.bool)
    mask[mask_indices] = True
    return mask

def evaluate_loss(model, data_loader, device):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)
    return total_loss / total_samples

def save_mask(mask_indices, out_dir="output"):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "selected_mask.npy"), np.array(mask_indices, dtype=np.int64))

def load_mask(mask_file):
    return np.load(mask_file)

def save_loss_history(loss_history, out_dir="output"):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "loss_history.npy"), np.array(loss_history))

def save_coreset_size(size, out_dir="output"):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "selected_coreset_size.txt"), "w") as f:
        f.write(str(size))

def save_accuracy(acc, out_dir="output"):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "accuracy.txt"), "w") as f:
        f.write(f"{acc:.4f}")