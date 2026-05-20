import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class FREEncoder(nn.Module):
    """
    Transformer‑based encoder that takes K (state, reward) pairs and outputs
    a latent vector z.  The input is treated as a set (no positional order).
    """
    def __init__(self, state_dim: int, reward_dim: int = 1,
                 d_model: int = 128, nhead: int = 4,
                 num_layers: int = 4, dim_feedforward: int = 256,
                 dropout: float = 0.1, latent_dim: int = 32):
        super().__init__()
        self.state_dim = state_dim
        self.reward_dim = reward_dim
        self.input_dim = state_dim + reward_dim
        self.d_model = d_model

        # Linear projection to d_model
        self.input_proj = nn.Linear(self.input_dim, d_model)

        # Transformer encoder
        encoder_layer = TransformerEncoderLayer(d_model, nhead,
                                                dim_feedforward,
                                                dropout, batch_first=True)
        self.transformer = TransformerEncoder(encoder_layer, num_layers)

        # Readout: mean pooling, then two heads for mean and logvar
        self.readout = nn.Linear(d_model, latent_dim * 2)

    def forward(self, states: torch.Tensor, rewards: torch.Tensor):
        """
        Args:
            states:  (B, K, state_dim)
            rewards: (B, K, 1)
        Returns:
            z_mean, z_logvar: each (B, latent_dim)
        """
        x = torch.cat([states, rewards], dim=-1)          # (B, K, input_dim)
        x = self.input_proj(x)                           # (B, K, d_model)
        x = self.transformer(x)                          # (B, K, d_model)
        x = x.mean(dim=1)                                # (B, d_model)
        h = self.readout(x)                               # (B, 2*latent_dim)
        z_mean, z_logvar = torch.chunk(h, 2, dim=-1)
        return z_mean, z_logvar

    def encode(self, states: torch.Tensor, rewards: torch.Tensor):
        """
        Sample from the Gaussian posterior.
        """
        z_mean, z_logvar = self.forward(states, rewards)
        std = torch.exp(0.5 * z_logvar)
        eps = torch.randn_like(std)
        z = z_mean + eps * std
        return z, z_mean, z_logvar