import torch
import torch.nn as nn
import math

class FREEncoder(nn.Module):
    """
    Permutation‑invariant transformer encoder.
    Input: K x (state_dim + reward_dim) tokens.
    Output: latent vector z of size latent_dim.
    """
    def __init__(self, state_dim=10, reward_dim=1, latent_dim=32, num_layers=2, nhead=4, dim_feedforward=128):
        super().__init__()
        self.token_dim = state_dim + reward_dim
        self.input_proj = nn.Linear(self.token_dim, dim_feedforward)
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_feedforward,
                                                   nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        # Mean & logvar for Gaussian latent
        self.fc_mean = nn.Linear(dim_feedforward, latent_dim)
        self.fc_logvar = nn.Linear(dim_feedforward, latent_dim)

    def forward(self, tokens):
        """
        tokens: (B, K, token_dim)
        """
        B, K, _ = tokens.shape
        x = self.input_proj(tokens)  # (B, K, d)
        x = self.transformer(x)      # (B, K, d)
        # pool across tokens
        x = x.transpose(1, 2)        # (B, d, K)
        x = self.pool(x).squeeze(-1)  # (B, d)
        mean = self.fc_mean(x)       # (B, latent_dim)
        logvar = self.fc_logvar(x)   # (B, latent_dim)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std         # reparameterization
        return z, mean, logvar