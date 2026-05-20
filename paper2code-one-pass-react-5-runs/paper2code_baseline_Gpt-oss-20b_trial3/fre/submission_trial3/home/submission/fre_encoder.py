import torch
import torch.nn as nn
import torch.nn.functional as F

class RewardEmbedding(nn.Module):
    """
    Embeds a scalar reward into a vector.
    """
    def __init__(self, reward_dim=32):
        super().__init__()
        self.linear = nn.Linear(1, reward_dim)

    def forward(self, reward):
        # reward: (batch, 1)
        return self.linear(reward)

class TransformerEncoder(nn.Module):
    """
    Permutation‑invariant encoder that takes a set of (state, reward) pairs
    and outputs a latent representation z.
    """
    def __init__(self, state_dim=2, reward_dim=32, latent_dim=64,
                 num_layers=2, num_heads=4, dim_feedforward=128):
        super().__init__()
        self.state_dim = state_dim
        self.reward_dim = reward_dim
        self.latent_dim = latent_dim

        self.token_embed = nn.Linear(state_dim + reward_dim, latent_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=latent_dim,
                                                   nhead=num_heads,
                                                   dim_feedforward=dim_feedforward)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output mean & logvar for reparameterization
        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim, latent_dim)

    def forward(self, states, rewards):
        """
        states: (B, K, state_dim)
        rewards: (B, K, 1)
        """
        B, K, _ = states.shape
        rewards_emb = RewardEmbedding()(rewards)  # (B, K, reward_dim)
        tokens = torch.cat([states, rewards_emb], dim=-1)  # (B, K, state_dim+reward_dim)
        tokens = self.token_embed(tokens)  # (B, K, latent_dim)
        # transformer expects (S, B, E)
        tokens = tokens.permute(1, 0, 2)
        out = self.transformer(tokens)  # (K, B, latent_dim)
        out = out.mean(dim=0)  # (B, latent_dim)
        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)
        return mu, logvar

class RewardDecoder(nn.Module):
    """
    Decodes reward for a state given latent z.
    """
    def __init__(self, state_dim=2, latent_dim=64, hidden_dims=[128, 128]):
        super().__init__()
        layers = []
        input_dim = state_dim + latent_dim
        for h in hidden_dims:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h
        layers.append(nn.Linear(input_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, states, z):
        """
        states: (B, N, state_dim)
        z: (B, latent_dim)
        """
        B, N, _ = states.shape
        z_exp = z.unsqueeze(1).expand(-1, N, -1)  # (B, N, latent_dim)
        inp = torch.cat([states, z_exp], dim=-1)  # (B, N, state_dim+latent_dim)
        out = self.net(inp).squeeze(-1)  # (B, N)
        return out

class FREModel(nn.Module):
    """
    Combines encoder and decoder.
    """
    def __init__(self, state_dim=2, reward_dim=32, latent_dim=64):
        super().__init__()
        self.encoder = TransformerEncoder(state_dim, reward_dim, latent_dim)
        self.decoder = RewardDecoder(state_dim, latent_dim)

    def encode(self, states, rewards):
        mu, logvar = self.encoder(states, rewards)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar

    def decode(self, states, z):
        return self.decoder(states, z)