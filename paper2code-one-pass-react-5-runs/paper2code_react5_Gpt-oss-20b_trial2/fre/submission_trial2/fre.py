import torch
import torch.nn as nn
import torch.nn.functional as F

class FREEncoder(nn.Module):
    """
    Transformer‑based encoder that maps a set of (state, reward) pairs to a latent vector.
    Implements the information‑bottleneck objective: p(z|context) ~ N(mean, var).
    """
    def __init__(self,
                 state_dim: int,
                 hidden_dim: int = 256,
                 num_layers: int = 4,
                 num_heads: int = 4,
                 latent_dim: int = 32):
        super().__init__()
        self.input_proj = nn.Linear(state_dim + 1, hidden_dim)  # +1 for scalar reward
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim,
                                                   nhead=num_heads,
                                                   dim_feedforward=hidden_dim*4,
                                                   dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                 num_layers=num_layers)
        self.mean_proj = nn.Linear(hidden_dim, latent_dim)
        self.logvar_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, states: torch.Tensor, rewards: torch.Tensor):
        """
        Args:
            states: [B, K, state_dim]
            rewards: [B, K, 1]
        Returns:
            z: [B, latent_dim]
            mean: [B, latent_dim]
            logvar: [B, latent_dim]
        """
        x = torch.cat([states, rewards], dim=-1)          # [B, K, state+1]
        x = self.input_proj(x)                            # [B, K, hidden]
        x = x.permute(1, 0, 2)                            # [K, B, hidden]
        x = self.transformer(x)                           # [K, B, hidden]
        x = x.mean(dim=0)                                 # [B, hidden]
        mean = self.mean_proj(x)
        logvar = self.logvar_proj(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z, mean, logvar

class FREDecoder(nn.Module):
    """
    MLP that predicts reward for a state given the latent vector z.
    """
    def __init__(self,
                 state_dim: int,
                 latent_dim: int,
                 hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, states: torch.Tensor, z: torch.Tensor):
        """
        Args:
            states: [B, K, state_dim]
            z: [B, latent_dim]
        Returns:
            preds: [B, K, 1]
        """
        B, K, _ = states.size()
        z_exp = z.unsqueeze(1).expand(-1, K, -1)          # [B, K, latent]
        x = torch.cat([states, z_exp], dim=-1)            # [B, K, state+latent]
        x = x.reshape(B * K, -1)
        out = self.net(x)
        return out.reshape(B, K, 1)