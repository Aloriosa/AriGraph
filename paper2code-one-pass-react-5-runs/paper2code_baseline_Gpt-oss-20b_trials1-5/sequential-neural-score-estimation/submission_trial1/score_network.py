import torch
import torch.nn as nn


class ScoreNetwork(nn.Module):
    """
    Simple MLP that estimates the conditional score
    ∇_θ log p_t(θ | x) for the forward variance‑exploding SDE.
    """

    def __init__(self, theta_dim: int, x_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.theta_emb = nn.Sequential(
            nn.Linear(theta_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.x_emb = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.t_emb = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
        )
        self.mlp = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, theta_dim),
        )

    def forward(self, theta: torch.Tensor, x: torch.Tensor, t: torch.Tensor):
        """
        Parameters
        ----------
        theta : Tensor of shape (B, d)
            Noised parameters θ_t.
        x : Tensor of shape (B, p)
            Observation vector.
        t : Tensor of shape (B, 1)
            Time scalar in [0, 1].
        """
        theta_e = self.theta_emb(theta)
        x_e = self.x_emb(x)
        t_e = self.t_emb(t)
        h = torch.cat([theta_e, x_e, t_e], dim=-1)
        return self.mlp(h)