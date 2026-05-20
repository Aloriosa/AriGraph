import torch
import torch.nn as nn
import math

class MLPScore(nn.Module):
    """
    Simple MLP that takes (θ_t, x, t) and outputs a score vector of dimension d.
    """
    def __init__(self, d_theta: int, d_x: int, hidden_dim: int = 256):
        super().__init__()
        self.d_theta = d_theta
        self.d_x = d_x
        # sinusoidal embedding for time t (scalar)
        self.time_emb = nn.Linear(1, 64)
        # embeddings for θ_t and x
        self.theta_emb = nn.Sequential(
            nn.Linear(d_theta, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.x_emb = nn.Sequential(
            nn.Linear(d_x, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        # final MLP
        self.net = nn.Sequential(
            nn.Linear(64 + hidden_dim*2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, d_theta),
        )

    def forward(self, theta: torch.Tensor,
                x: torch.Tensor,
                t: torch.Tensor) -> torch.Tensor:
        """
        theta: (B, d_theta)
        x:     (B, d_x)
        t:     (B, 1)   (time in [0,1])
        """
        theta_e = self.theta_emb(theta)
        x_e = self.x_emb(x)
        t_e = self.time_emb(t)
        inp = torch.cat([theta_e, x_e, t_e], dim=-1)
        return self.net(inp)