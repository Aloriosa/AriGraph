# Small utility functions used by the training and sampling scripts.

import math
import torch


def alpha_beta(t):
    """
    Linear interpolation coefficients.
    Returns alpha_t, beta_t, dot_alpha_t, dot_beta_t
    """
    alpha = 1.0 - t
    beta = t
    dot_alpha = -1.0 * torch.ones_like(t)
    dot_beta = torch.ones_like(t)
    return alpha, beta, dot_alpha, dot_beta


def get_timestep_embedding(t, embedding_dim):
    """
    Builds sinusoidal embeddings from a scalar time t.
    t: Tensor of shape [batch]
    embedding_dim: int
    Returns Tensor of shape [batch, embedding_dim]
    """
    device = t.device
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=device) * -emb)
    emb = t[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    return emb