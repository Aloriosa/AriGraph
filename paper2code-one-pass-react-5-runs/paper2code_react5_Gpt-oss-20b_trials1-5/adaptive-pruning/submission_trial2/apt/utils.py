"""
Utility functions for salience calculation, kurtosis, and metric logging.
"""

import math
import torch
import numpy as np
import torch.nn.functional as F
from typing import Dict, List, Tuple

def compute_head_salience(attn_module, grads: Dict, salience_dict: Dict):
    """
    Compute a simple salience score for each attention head.
    Score = |weight| * |gradient| summed over the head's rows.
    """
    # attn_module.q_proj is a LoRALinear; we look at its weight
    # The weight shape is (out_features, in_features)
    weight = attn_module.q_proj.weight  # frozen
    grad = grads.get(weight, None)
    if grad is None:
        # No gradient (e.g., if not part of loss), skip
        return

    # head dimension
    head_dim = weight.shape[1] // attn_module.num_heads
    # Split weight into heads
    weight_heads = weight.view(attn_module.num_heads, head_dim, -1)  # (heads, head_dim, in_dim)
    grad_heads = grad.view(attn_module.num_heads, head_dim, -1)

    # Sum over head_dim and in_dim
    salience = torch.sum(torch.abs(weight_heads) * torch.abs(grad_heads), dim=(1, 2))
    # salience: tensor of shape (num_heads,)
    return salience.cpu().numpy()

def kurtosis(x: torch.Tensor):
    """
    Compute kurtosis for a 1‑D tensor: (E[(x-μ)^4]/σ^4) - 3
    """
    n = x.numel()
    if n < 4:
        return torch.tensor(0.0)
    mean = torch.mean(x)
    std = torch.std(x, unbiased=False)
    if std.item() == 0:
        return torch.tensor(0.0)
    m4 = torch.mean((x - mean) ** 4)
    return m4 / (std ** 4) - 3

def add_kurtosis_to_salience(salience: np.ndarray, activations: torch.Tensor):
    """
    Augment salience with sqrt(kurtosis) of the head activations.
    """
    # activations shape: (batch, seq_len, heads, head_dim)
    # Compute mean activation per head
    head_mean = activations.mean(dim=(0, 1, 3))  # (heads,)
    # Compute kurtosis per head
    k = torch.tensor([kurtosis(head_mean[i].unsqueeze(0)) for i in range(len(head_mean))])
    # Add sqrt(kurtosis) to salience
    return salience + np.sqrt(k.cpu().numpy())

def log_metrics(metrics: Dict, prefix: str = ""):
    """
    Pretty‑print a dictionary of metrics.
    """
    for k, v in metrics.items():
        print(f"{prefix}{k}: {v}")