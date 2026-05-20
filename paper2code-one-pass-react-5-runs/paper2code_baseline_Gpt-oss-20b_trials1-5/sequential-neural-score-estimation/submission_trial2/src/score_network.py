import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import sinusoidal_embedding

class MLPScoreNet(nn.Module):
    """
    Time‑dependent score network s(θ_t, x, t) → ∇_θ log p_t(θ_t | x).
    Architecture: [θ_emb | x_emb | t_emb] → 3‑layer MLP → d outputs.
    """
    def __init__(self, dim_theta: int, dim_x: int, hidden: int = 256, t_emb: int = 64):
        super().__init__()
        self.dim_theta = dim_theta
        self.dim_x = dim_x
        self.t_emb = t_emb

        # Embedding networks
        self.theta_emb = nn.Sequential(
            nn.Linear(dim_theta, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU()
        )
        self.x_emb = nn.Sequential(
            nn.Linear(dim_x, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU()
        )
        # Final MLP
        self.net = nn.Sequential(
            nn.Linear(hidden * 3, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim_theta)
        )

    def forward(self, theta_t: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            theta_t: (B, d)   current perturbed parameter
            x:       (B, p)   simulated data
            t:       (B,)     time scalar in [0,1]
        Returns:
            score:   (B, d)
        """
        theta_e = self.theta_emb(theta_t)
        x_e = self.x_emb(x)
        t_e = sinusoidal_embedding(t, self.t_emb)
        h = torch.cat([theta_e, x_e, t_e], dim=-1)
        return self.net(h)