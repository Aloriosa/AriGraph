import torch
import torch.nn.functional as F
import random
import numpy as np

def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_random_mapping(num_pretrained: int, num_target: int, seed: int = 42):
    """
    Create a random injective mapping from a subset of pretrained classes
    to target classes. Returns a dict mapping pretrained idx -> target idx.
    """
    set_seed(seed)
    pretrained_subset = random.sample(range(num_pretrained), num_target)
    mapping = {p: t for p, t in zip(pretrained_subset, range(num_target))}
    return mapping

def apply_mapping(logits, mapping):
    """
    logits: Tensor of shape (B, num_pretrained)
    mapping: dict pretrained_idx -> target_idx
    Returns logits of shape (B, num_target)
    """
    device = logits.device
    num_target = max(mapping.values()) + 1
    mapped = torch.zeros(logits.size(0), num_target, device=device)
    for p_idx, t_idx in mapping.items():
        mapped[:, t_idx] = logits[:, p_idx]
    return mapped

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(logits, 1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        logits = model(imgs)
        _, preds = torch.max(logits, 1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
    return correct / total