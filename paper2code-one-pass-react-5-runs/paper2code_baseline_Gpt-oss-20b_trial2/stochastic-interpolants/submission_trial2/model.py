"""
Minimal velocity network for stochastic interpolants.
The network predicts the velocity field b_t(x) given (x, t).
"""

import math
import torch
import torch.nn as nn


def time_embedding(t, dim=16, device=None):
    """
    Sinusoidal time embedding (like in Transformers).
    t: scalar or tensor of shape (B,)
    dim: dimension of the embedding (must be even)
    Returns: tensor of shape (B, dim)
    """
    if device is None:
        device = t.device
    t = t.to(device)
    # Create a (B, dim//2) matrix of frequencies
    half = dim // 2
    freq = 10000 ** (torch.arange(half, device=device, dtype=t.dtype) / half)
    angles = t.unsqueeze(1) * freq  # (B, half)
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
    return emb  # (B, dim)


class VelocityNet(nn.Module):
    """
    Simple MLP: input = [x; time_embed], output = velocity b_t(x)
    """

    def __init__(self, x_dim=784, time_emb_dim=16, hidden_dim=512):
        super().__init__()
        in_dim = x_dim + time_emb_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, x_dim),
        )

    def forward(self, x, t):
        """
        x: (B, x_dim)
        t: (B,) scalar time in [0, 1]
        """
        t_emb = time_embedding(t, dim=self.net[0].in_features - x.shape[1])
        inp = torch.cat([x, t_emb], dim=1)
        return self.net(inp)