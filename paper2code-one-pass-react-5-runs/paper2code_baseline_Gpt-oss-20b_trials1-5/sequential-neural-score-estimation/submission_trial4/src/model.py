import torch
import torch.nn as nn
import torch.nn.functional as F

class ScoreMLP(nn.Module):
    """
    Simple MLP that takes concatenated [theta_t, x, t_emb] and outputs a score vector.
    """
    def __init__(self, dim_theta=2, dim_x=2, hidden=256, n_layers=3):
        super().__init__()
        self.dim_theta = dim_theta
        self.dim_x = dim_x
        self.t_emb_dim = 64
        self.hidden = hidden
        self.n_layers = n_layers

        # Embedding for time t
        self.t_embed = nn.Linear(self.t_emb_dim, hidden)

        # Linear embedding for theta and x
        self.theta_embed = nn.Linear(dim_theta, hidden)
        self.x_embed = nn.Linear(dim_x, hidden)

        # MLP
        layers = []
        for _ in range(n_layers):
            layers.append(nn.Linear(hidden, hidden))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(hidden, dim_theta))
        self.mlp = nn.Sequential(*layers)

    def forward(self, theta_t, x, t):
        """
        Args:
            theta_t: Tensor (batch, dim)
            x: Tensor (batch, dim)
            t: Tensor (batch,)
        Returns:
            score: Tensor (batch, dim)
        """
        # Time embedding
        t_emb = self.positional_encoding(t)  # (batch, t_emb_dim)
        t_emb = self.t_embed(t_emb)          # (batch, hidden)

        # Input embeddings
        theta_emb = self.theta_embed(theta_t)
        x_emb = self.x_embed(x)

        # Concatenate
        h = torch.cat([theta_emb, x_emb, t_emb], dim=1)
        score = self.mlp(h)
        return score

    @staticmethod
    def positional_encoding(t, emb_dim=64):
        """
        Sinusoidal positional encoding as in Transformers.
        t: Tensor (batch,)
        Returns: Tensor (batch, emb_dim)
        """
        device = t.device
        position = t[:, None]  # (batch, 1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2, device=device) * (-torch.log(torch.tensor(10000.0)) / emb_dim))
        pe = torch.zeros_like(t[:, None].repeat(1, emb_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe