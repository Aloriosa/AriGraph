import torch
import torch.nn.functional as F
import random
import numpy as np
from einops import rearrange

def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def random_mask(batch_size: int, height: int, width: int, p: float = 0.3):
    """Generate a random binary mask of shape (B, 1, H, W)."""
    mask = torch.bernoulli(p * torch.ones(batch_size, 1, height, width))
    return mask

def get_time_embedding(t, num_channels=32):
    """
    Create a sinusoidal time embedding (as in transformers) and
    return a tensor of shape (B, C, 1, 1) that can be broadcasted
    to match image spatial dimensions.
    """
    device = t.device
    half = num_channels // 2
    emb = torch.arange(half, dtype=torch.float32, device=device)
    emb = 10000 ** (emb / half)
    emb = t[:, None] / emb
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    return emb[:, :, None, None]  # (B, C, 1, 1)