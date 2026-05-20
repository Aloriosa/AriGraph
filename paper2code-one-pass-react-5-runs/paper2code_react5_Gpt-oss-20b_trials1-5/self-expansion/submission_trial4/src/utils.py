import random
import numpy as np
import torch

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def accuracy(output: torch.Tensor, target: torch.Tensor, classes: list):
    """Compute accuracy for a subset of classes."""
    logits = output[:, classes]
    preds = logits.argmax(dim=1)
    return (preds == target).float().mean().item()

def compute_mean_std(values: torch.Tensor):
    """Compute mean and std of a tensor along the first dimension."""
    mean = values.mean()
    std = values.std(unbiased=False) + 1e-8
    return mean, std