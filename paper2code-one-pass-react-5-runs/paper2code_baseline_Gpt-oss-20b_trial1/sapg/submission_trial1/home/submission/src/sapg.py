"""
Simplified Split & Aggregate Policy Gradient (SAPG) implementation.
The code follows the high‑level algorithm in the paper but is adapted to
a CPU‑based Gymnasium environment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple

# ---------- Policy network ---------------------------------------------

class SAPGPolicy(nn.Module):
    """
    Shared backbone + per‑policy latent vector (phi).
    The policy outputs parameters of a diagonal Gaussian.
    """
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 64,
                 latent_dim: int = 16, entropy_coef: float = 0.0):
        super().__init__()
        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # Per‑policy latent vector (learnable)
        self.phi = nn.Parameter(torch.randn(latent_dim))
        # Output layers
        self.mu_head = nn.Linear(hidden_dim + latent_dim, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))  # fixed std

        self.entropy_coef = entropy_coef

    def forward(self, obs: torch.Tensor):
        """
        Compute mu and log_std for a batch of observations.
        """
        h = self.backbone(obs)
        # broadcast phi to batch
        phi = self.phi.unsqueeze(0).expand_as(h)
        h = torch.cat([h, phi], dim=-1)
        mu = self.mu_head(h)
        log_std = self.log_std.expand_as(mu)
        return mu, log_std

    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        """
        Sample an action given observation.
        """
        mu, log_std = self.forward(obs)
        std = log_std.exp()
        if deterministic:
            action = mu
        else:
            action = mu + torch.randn_like(mu) * std
        return action.clamp(-1.0, 1.0)

    def sample_action_and_logp(self, obs: torch.Tensor):
        """
        Sample action and return log probability.
        """
        mu, log_std = self.forward(obs)
        std = log_std.exp()
        eps = torch.randn_like(mu)
        action = mu + eps * std
        logp = (-0.5 * ((eps) ** 2 + 2 * log_std + np.log(2 * np.pi))
                ).sum(dim=-1)
        return action.clamp(-1.0, 1.0), logp

    def compute_logp(self, obs: torch.Tensor, actions: torch.Tensor):
        """
        Compute log probability of given actions under current policy.
        """
        mu, log_std = self.forward(obs)
        std = log_std.exp()
        var = std ** 2
        logp = -0.5 * (((actions - mu) ** 2) / var + 2 * log_std + np.log(2 * np.pi))
        return logp.sum(dim=-1)