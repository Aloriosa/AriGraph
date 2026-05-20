import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalEmbedding(nn.Module):
    """Sinusoidal embedding of a scalar time t."""
    def __init__(self, embed_dim=64):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, t):
        """
        Args:
            t: Tensor of shape (batch, 1)
        Returns:
            Tensor of shape (batch, embed_dim)
        """
        device = t.device
        half_dim = self.embed_dim // 2
        emb = torch.arange(half_dim, dtype=torch.float, device=device)
        emb = torch.pow(10000, -2 * emb / self.embed_dim)
        emb = t * emb
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb


class MLPEmbedding(nn.Module):
    """MLP embedding for a vector input."""
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class ScoreNetwork(nn.Module):
    """
    Conditional score network s(theta_t, x, t) -> grad_theta log p_t(theta_t | x)
    """
    def __init__(self, theta_dim, x_dim, hidden_dim=256, t_embed_dim=64):
        super().__init__()
        self.theta_emb = MLPEmbedding(theta_dim, max(30, 4 * theta_dim), hidden_dim)
        self.x_emb = MLPEmbedding(x_dim, max(30, 4 * x_dim), hidden_dim)
        self.t_emb = SinusoidalEmbedding(t_embed_dim)
        # final MLP
        self.net = nn.Sequential(
            nn.Linear(self.theta_emb.net[-1].out_features +
                      self.x_emb.net[-1].out_features +
                      t_embed_dim,
                      hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, theta_dim)
        )

    def forward(self, theta_t, x, t):
        """
        Args:
            theta_t: (B, theta_dim)
            x: (B, x_dim)
            t: (B, 1) values in [0,1]
        Returns:
            grad_theta log p_t(theta_t | x)   (B, theta_dim)
        """
        theta_e = self.theta_emb(theta_t)
        x_e = self.x_emb(x)
        t_e = self.t_emb(t)
        inp = torch.cat([theta_e, x_e, t_e], dim=1)
        return self.net(inp)