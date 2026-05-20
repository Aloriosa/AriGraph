"""
Utility functions for masks, Fourier embeddings, etc.
"""

import numpy as np
import torch


def random_attention_mask(seq_len, p=0.3):
    """
    Generate a random binary attention mask (seq_len x seq_len) with sparsity p.
    """
    mask = torch.bernoulli(p * torch.ones(seq_len, seq_len))
    mask.fill_diagonal_(1.0)  # self‑attention always allowed
    return mask.bool()


def identity_attention_mask(seq_len):
    """
    Identity mask (only self‑attention).
    """
    return torch.eye(seq_len, dtype=torch.bool)


def full_attention_mask(seq_len):
    """
    Full (dense) mask.
    """
    return torch.ones(seq_len, seq_len, dtype=torch.bool)


def fourier_features(x, dim):
    """
    Random Fourier features for a scalar or 1‑d array x.
    """
    # x: [batch, ...] or scalar
    # dim: number of Fourier dimensions
    omega = np.random.normal(size=(dim,)).astype(np.float32)
    b = np.random.uniform(0, 2 * np.pi, size=(dim,)).astype(np.float32)
    if isinstance(x, torch.Tensor):
        x = x.unsqueeze(-1)  # [batch, 1]
        return torch.cos(x @ torch.tensor(omega, device=x.device) + torch.tensor(b, device=x.device))
    else:
        return np.cos(x * omega + b)