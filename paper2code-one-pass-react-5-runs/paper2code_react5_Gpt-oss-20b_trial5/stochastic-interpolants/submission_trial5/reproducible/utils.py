import math
import random
import numpy as np
import torch

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_time_embedding(t: torch.Tensor, dim: int = 32) -> torch.Tensor:
    """
    Positional (sinusoidal) time embedding as used in Diffusion models.
    """
    batch = t.shape[0]
    device = t.device
    half = dim // 2
    emb = math.log(10000.0) / (half - 1)
    emb = torch.exp(torch.arange(half, device=device) * -emb)
    emb = t[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    return emb  # shape: [B, dim]