import torch
import torch.nn as nn
from src.utils import MLP, sinusoidal_embedding


class ScoreNetwork(nn.Module):
    """
    Conditional score network s(θ_t, x, t) ≈ ∇_θ log p_t(θ_t | x)
    """
    def __init__(self, theta_dim=1, x_dim=1, hidden=256):
        super().__init__()
        # Embedding dimensions
        self.theta_emb = MLP(theta_dim, max(30, 4 * theta_dim), hidden)
        self.x_emb = MLP(x_dim, max(30, 4 * x_dim), hidden)
        # Final network
        self.net = MLP(
            max(30, 4 * theta_dim) + max(30, 4 * x_dim) + 64,
            theta_dim,
            hidden
        )

    def forward(self, theta, x, t):
        """
        theta: (B, d)
        x: (B, p)
        t: (B,) scalar in [0,1]
        """
        theta_e = self.theta_emb(theta)
        x_e = self.x_emb(x)
        t_e = sinusoidal_embedding(t)  # (B, 64)
        h = torch.cat([theta_e, x_e, t_e], dim=1)
        return self.net(h)