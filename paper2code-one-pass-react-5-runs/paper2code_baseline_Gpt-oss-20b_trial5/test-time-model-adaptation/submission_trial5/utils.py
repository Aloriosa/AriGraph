"""
Utility functions for the FOA implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def entropy(preds: torch.Tensor) -> torch.Tensor:
    """
    Compute the entropy of a probability distribution.

    Args:
        preds: Tensor of shape (batch, num_classes) with probabilities.

    Returns:
        Tensor of shape (batch,) containing entropy values per sample.
    """
    eps = 1e-12
    log_probs = torch.log(preds + eps)
    ent = -torch.sum(preds * log_probs, dim=1)
    return ent


def to_device(tensor, device):
    return tensor.to(device)


def clip_tensor(tensor, min_val=-10.0, max_val=10.0):
    return torch.clamp(tensor, min=min_val, max=max_val)