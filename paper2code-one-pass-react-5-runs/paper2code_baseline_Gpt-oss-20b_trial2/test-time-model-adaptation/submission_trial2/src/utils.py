import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm


def entropy(preds: torch.Tensor) -> torch.Tensor:
    """
    Compute per‑sample entropy of a probability distribution.
    preds: (B, C) logits
    """
    probs = F.softmax(preds, dim=-1)
    log_probs = torch.log(probs + 1e-12)
    return -torch.sum(probs * log_probs, dim=-1)  # (B,)


def mean_std(tensors: torch.Tensor, dim: int = 0):
    """
    Compute mean and std over the first dimension.
    tensors: (B, D)
    """
    return torch.mean(tensors, dim=dim), torch.std(tensors, dim=dim)


def accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute classification accuracy.
    preds: (B, C) logits
    targets: (B,)
    """
    _, pred_cls = torch.max(preds, dim=1)
    correct = pred_cls.eq(targets).sum().item()
    return correct / preds.size(0)


def moving_average(prev: torch.Tensor, current: torch.Tensor, alpha: float):
    """
    Exponential moving average: new = alpha * current + (1 - alpha) * prev
    """
    return alpha * current + (1 - alpha) * prev