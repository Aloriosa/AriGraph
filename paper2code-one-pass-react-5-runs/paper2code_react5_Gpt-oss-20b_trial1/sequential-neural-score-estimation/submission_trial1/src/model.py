"""
Conditional score network used in TSNPSE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    """
    Sinusoidal positional embedding for time t.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t : (B,) or scalar
        returns : (B, dim)
        """
        device = t.device
        half_dim = self.dim // 2
        emb = torch.exp(
            torch.arange(half_dim, dtype=torch.float32, device=device)
            * (-torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1))
        )
        emb = t.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class ScoreMLP(nn.Module):
    """
    3‑layer MLP that takes (θ_t, x, t) and outputs the posterior score.
    """
    def __init__(self, dim_theta: int, dim_x: int):
        super().__init__()
        self.theta_embed = nn.Sequential(
            nn.Linear(dim_theta, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(),
            nn.Linear(256, 256)
        )
        self.x_embed = nn.Sequential(
            nn.Linear(dim_x, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(),
            nn.Linear(256, 256)
        )
        self.t_embed = SinusoidalPosEmb(64)

        self.net = nn.Sequential(
            nn.Linear(256 * 3, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(),
            nn.Linear(256, dim_theta)  # output dimension = dim_theta
        )

    def forward(self, theta_t: torch.Tensor, x: torch.Tensor, t: torch.Tensor):
        """
        theta_t : (B, dim_theta)
        x       : (B, dim_x)
        t       : (B,) or scalar
        """
        theta_emb = self.theta_embed(theta_t)
        x_emb = self.x_embed(x)
        t_emb = self.t_embed(t)
        h = torch.cat([theta_emb, x_emb, t_emb], dim=-1)
        return self.net(h)