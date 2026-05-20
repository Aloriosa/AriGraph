import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MLPEmbedding(nn.Module):
    """
    Simple MLP embedding for a single vector.
    """
    def __init__(self, input_dim, hidden_dim=256, num_layers=3):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            in_dim = hidden_dim
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class SinusoidalEmbedding(nn.Module):
    """
    Sinusoidal embedding of a scalar time t in [0, 1].
    """
    def __init__(self, dim=64):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t shape: (N, 1)
        device = t.device
        div_term = torch.exp(torch.arange(0, self.dim, 2, device=device,
                                          dtype=torch.float32) *
                             -(np.log(10000.0) / self.dim))
        t = t.unsqueeze(-1)  # (N,1,1)
        sin = torch.sin(t * div_term)  # (N, dim/2)
        cos = torch.cos(t * div_term)
        return torch.cat([sin, cos], dim=-1)  # (N, dim)

class ScoreNetwork(nn.Module):
    """
    Conditional score network: s(θ_t, x, t) ≈ ∇_θ log p_t(θ_t | x)
    """
    def __init__(self, dim_theta, dim_x, dim_t_emb=64, hidden_dim=256):
        super().__init__()
        self.theta_emb = MLPEmbedding(dim_theta, hidden_dim)
        self.x_emb = MLPEmbedding(dim_x, hidden_dim)
        self.t_emb = SinusoidalEmbedding(dim_t_emb)
        # Final MLP
        self.final = MLPEmbedding(hidden_dim * 3, hidden_dim, num_layers=3)
        self.out = nn.Linear(hidden_dim, dim_theta)

    def forward(self, theta_t, x, t):
        """
        theta_t: (N, d)
        x: (d,) broadcasted or (N,d)
        t: (N,) scalar times
        """
        theta_e = self.theta_emb(theta_t)
        # Ensure x has same shape
        if x.dim() == 1:
            x = x.expand_as(theta_t)
        x_e = self.x_emb(x)
        t_e = self.t_emb(t.unsqueeze(-1))
        h = torch.cat([theta_e, x_e, t_e], dim=-1)
        h = self.final(h)
        return self.out(h)