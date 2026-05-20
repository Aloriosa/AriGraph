# utils.py
import torch
import numpy as np
from torchvision import transforms

def create_random_mask(shape, p=0.3):
    """
    Create a binary mask with probability p for 1 (keep) and 0 for 0 (masked).
    shape: (H, W)
    """
    mask = torch.bernoulli(p * torch.ones(shape)).type(torch.float32)
    return mask

def sample_base(x1, mask, noise_std=0.1):
    """
    Given target image x1 and binary mask, create a base sample
    x0 = mask * x1 + (1 - mask) * noise
    noise ~ N(0, noise_std^2)
    """
    noise = torch.randn_like(x1) * noise_std
    return mask * x1 + (1 - mask) * noise

def time_embedding(t, dim=64):
    """
    Sinusoidal time embedding (like in transformers).
    t: Tensor of shape (batch,)
    dim: embedding dimension
    Returns: Tensor of shape (batch, dim)
    """
    device = t.device
    half_dim = dim // 2
    emb = np.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
    emb = t[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    return emb  # shape (batch, dim)