"""
Functional Reward Encoding (FRE) – encoder & decoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    """
    Permutation‑invariant transformer encoder.
    Takes K context (state, reward) pairs and outputs a latent vector z.
    """
    def __init__(self, state_dim, reward_dim=1, latent_dim=32, n_layers=4, n_heads=4, d_ff=256):
        super().__init__()
        self.state_dim = state_dim
        self.reward_dim = reward_dim
        self.latent_dim = latent_dim

        input_dim = state_dim + reward_dim
        self.input_proj = nn.Linear(input_dim, d_ff)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_ff,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output mean & logvar for Gaussian latent
        self.mu_layer = nn.Linear(d_ff, latent_dim)
        self.logvar_layer = nn.Linear(d_ff, latent_dim)

    def forward(self, states, rewards):
        """
        states: (B, K, D)
        rewards: (B, K, 1)
        """
        x = torch.cat([states, rewards], dim=-1)          # (B, K, D+1)
        x = self.input_proj(x)                           # (B, K, d_ff)
        # Transformer expects (B, K, d_ff)
        x = self.transformer(x)                          # (B, K, d_ff)
        # Pool (mean)
        x = x.mean(dim=1)                                 # (B, d_ff)
        mu = self.mu_layer(x)                             # (B, latent_dim)
        logvar = self.logvar_layer(x)                     # (B, latent_dim)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar

class Decoder(nn.Module):
    """
    Predicts reward for a single state given latent z.
    """
    def __init__(self, state_dim, latent_dim, d_ff=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + latent_dim, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, 1)
        )

    def forward(self, states, z):
        """
        states: (B, S, D)
        z: (B, latent_dim)
        """
        B, S, D = states.shape
        z_exp = z.unsqueeze(1).expand(-1, S, -1)          # (B, S, latent_dim)
        x = torch.cat([states, z_exp], dim=-1)            # (B, S, D+latent_dim)
        out = self.net(x)                                 # (B, S, 1)
        return out.squeeze(-1)                            # (B, S)