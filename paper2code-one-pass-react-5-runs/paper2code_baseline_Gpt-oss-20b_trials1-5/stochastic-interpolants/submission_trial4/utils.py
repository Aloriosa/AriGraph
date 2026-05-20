import math
import torch

def sinusoidal_embedding(t, dim=64):
    """
    Return a sinusoidal embedding of time t∈[0,1] of dimension `dim`.
    """
    device = t.device
    half_dim = dim // 2
    frequencies = torch.exp(
        torch.arange(half_dim, dtype=torch.float32, device=device) *
        -(math.log(10000.0) / (half_dim - 1))
    )
    args = t.unsqueeze(-1) * frequencies * 2 * math.pi
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:  # pad if odd
        emb = torch.cat([emb, torch.zeros_like(emb[..., :1])], dim=-1)
    return emb

def alpha_beta(t):
    """Simple linear coefficients: α_t = 1 - t, β_t = t."""
    return 1.0 - t, t

def dot_alpha_beta(t):
    """Derivatives: dα/dt = -1, dβ/dt = 1."""
    return -1.0, 1.0