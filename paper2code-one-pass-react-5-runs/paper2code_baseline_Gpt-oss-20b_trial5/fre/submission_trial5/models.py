#!/usr/bin/env python3
"""
PyTorch modules used in the FRE implementation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- Reward‑function encoder ----------
class RewardEncoder(nn.Module):
    """
    Transformer‑based encoder that maps a set of (state, reward) pairs
    to a latent vector z ∈ ℝ^d.
    """
    def __init__(
        self,
        state_dim: int,
        reward_dim: int = 1,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        latent_dim: int = 32,
    ):
        super().__init__()
        # Input embedding: state + reward
        self.input_proj = nn.Linear(state_dim + reward_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Map transformer output to Gaussian parameters
        self.mean_head = nn.Linear(embed_dim, latent_dim)
        self.logvar_head = nn.Linear(embed_dim, latent_dim)

    def forward(self, states, rewards):
        """
        states: Tensor [B, K, state_dim]
        rewards: Tensor [B, K, 1]
        Returns:
            z: Tensor [B, latent_dim] – sampled latent vector
            (mean, logvar) for KL term
        """
        x = torch.cat([states, rewards], dim=-1)  # [B, K, state_dim+1]
        x = self.input_proj(x)  # [B, K, embed_dim]
        x = self.transformer(x)  # [B, K, embed_dim]
        x = x.mean(dim=1)  # [B, embed_dim]
        mean = self.mean_head(x)
        logvar = self.logvar_head(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z, mean, logvar


# ---------- Q‑network ----------
class QNetwork(nn.Module):
    """
    Q(s, a, z) – conditioned on state, action, and latent reward encoding.
    For CartPole, actions are discrete (0 or 1), so we use a simple linear head.
    """
    def __init__(
        self,
        state_dim: int,
        latent_dim: int,
        hidden_dim: int = 128,
        n_actions: int = 2,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, state, z):
        """
        state: [B, state_dim]
        z: [B, latent_dim]
        Returns Q-values for all actions: [B, n_actions]
        """
        x = torch.cat([state, z], dim=-1)
        return self.net(x)