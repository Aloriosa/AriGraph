import torch
import torch.nn as nn
import torch.nn.functional as F

class RewardEmbedding(nn.Module):
    """Embed a scalar reward into a vector."""
    def __init__(self, embed_dim: int = 32):
        super().__init__()
        self.linear = nn.Linear(1, embed_dim)

    def forward(self, reward: torch.Tensor):
        # reward: (B, K, 1)
        return self.linear(reward)

class FreEncoder(nn.Module):
    """
    Transformer‑based VAE encoder that maps a set of (state, reward) pairs
    to a latent vector z ∈ ℝ^latent_dim.
    """
    def __init__(self, state_dim: int, latent_dim: int = 64,
                 n_heads: int = 4, n_layers: int = 4,
                 embed_dim: int = 128):
        super().__init__()
        self.state_embed = nn.Linear(state_dim, embed_dim)
        self.reward_embed = RewardEmbedding(embed_dim)
        self.proj = nn.Linear(2 * embed_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.mean_head = nn.Linear(embed_dim, latent_dim)
        self.std_head = nn.Linear(embed_dim, latent_dim)

    def forward(self, states: torch.Tensor, rewards: torch.Tensor):
        """
        states: (B, K, state_dim)
        rewards: (B, K, 1)
        Returns: mean (B, latent_dim), std (B, latent_dim)
        """
        state_emb = self.state_embed(states)          # (B, K, embed_dim)
        reward_emb = self.reward_embed(rewards)       # (B, K, embed_dim)
        x = torch.cat([state_emb, reward_emb], dim=-1)  # (B, K, 2*embed_dim)
        x = self.proj(x)  # (B, K, embed_dim)
        x = self.transformer(x)  # (B, K, embed_dim)
        x = x.transpose(1, 2)  # (B, embed_dim, K)
        x = self.pool(x).squeeze(-1)  # (B, embed_dim)
        mean = self.mean_head(x)  # (B, latent_dim)
        std = F.softplus(self.std_head(x)) + 1e-5  # ensure positivity
        return mean, std

    def sample_z(self, states: torch.Tensor, rewards: torch.Tensor):
        mean, std = self.forward(states, rewards)
        eps = torch.randn_like(mean)
        return mean + eps * std

class FreDecoder(nn.Module):
    """
    Decoder that predicts reward for a single state given latent z.
    """
    def __init__(self, state_dim: int, latent_dim: int = 64, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state: torch.Tensor, z: torch.Tensor):
        """
        state: (B, state_dim)
        z:    (B, latent_dim)
        """
        x = torch.cat([state, z], dim=-1)
        return self.net(x)  # (B, 1)