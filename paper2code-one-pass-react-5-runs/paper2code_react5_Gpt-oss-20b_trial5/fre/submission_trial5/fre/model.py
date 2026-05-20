import torch
import torch.nn as nn
import torch.nn.functional as F

class FREEncoder(nn.Module):
    """
    Simple MLP‑based encoder that maps a set of (state, reward) pairs
    to a latent vector z ~ N(mean, sigma^2).
    """
    def __init__(self, state_dim, reward_dim=1,
                 hidden_dim=256, latent_dim=32):
        super().__init__()
        self.input_dim = state_dim + reward_dim
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mean_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

    def forward(self, states, rewards):
        """
        states: (batch, K, state_dim)
        rewards: (batch, K, 1)
        """
        x = torch.cat([states, rewards], dim=-1)          # (B, K, D)
        B, K, _ = x.shape
        x = x.reshape(B*K, -1)                           # (B*K, D)
        x = self.mlp(x)                                  # (B*K, H)
        x = x.reshape(B, K, -1)                          # (B, K, H)
        x = x.mean(dim=1)                                # (B, H)
        mean = self.mean_head(x)                          # (B, Z)
        logvar = self.logvar_head(x)                      # (B, Z)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z, mean, logvar


class FREDecoder(nn.Module):
    """
    Decodes reward values for a batch of states given a latent z.
    """
    def __init__(self, state_dim, latent_dim=32,
                 hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, states, z):
        """
        states: (batch, K, state_dim)
        z:     (batch, latent_dim)
        """
        B, K, _ = states.shape
        z_exp = z.unsqueeze(1).expand(-1, K, -1)          # (B, K, Z)
        x = torch.cat([states, z_exp], dim=-1)           # (B, K, D+Z)
        x = x.reshape(B*K, -1)                           # (B*K, D+Z)
        out = self.net(x).squeeze(-1)                    # (B*K)
        out = out.reshape(B, K)                          # (B, K)
        return out