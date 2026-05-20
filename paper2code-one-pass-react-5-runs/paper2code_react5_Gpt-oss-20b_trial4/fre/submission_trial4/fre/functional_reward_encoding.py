"""
Functional Reward Encoding (FRE) implementation.

This module implements the encoder/decoder architecture described in the paper.
The encoder is a transformer that consumes a set of (state, reward) pairs
and produces a latent distribution p(z | context).  The decoder predicts
the reward for a new state given the latent code.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List


class RewardPrior:
    """
    Sample random reward functions from a mixture of
    goal‑reaching, linear, and MLP types.
    """

    def __init__(self, state_dim: int, device: torch.device):
        self.state_dim = state_dim
        self.device = device
        self.mlp_dim = 32
        # Pre‑generate random MLP weights
        self.mlp_weights = [
            (
                torch.randn(state_dim, self.mlp_dim, device=device) * 0.5,
                torch.randn(self.mlp_dim, 1, device=device) * 0.5,
            )
            for _ in range(5)
        ]

    def sample(self) -> Tuple[callable, str]:
        """Return a reward function and its type."""
        rtype = random.choice(["goal", "linear", "mlp"])
        if rtype == "goal":
            goal = torch.randn(self.state_dim, device=self.device)

            def fn(states: torch.Tensor) -> torch.Tensor:
                # states: (N, state_dim)
                dist = torch.norm(states - goal, dim=-1)
                return torch.where(dist > 0.2,
                                   torch.full_like(dist, -1.0),
                                   torch.full_like(dist, 0.0))

            return fn, "goal"

        elif rtype == "linear":
            w = torch.randn(self.state_dim, device=self.device)
            b = torch.randn(1, device=self.device)

            def fn(states: torch.Tensor) -> torch.Tensor:
                return (states @ w.unsqueeze(-1)).squeeze(-1) + b

            return fn, "linear"

        else:
            w1, w2 = random.choice(self.mlp_weights)

            def fn(states: torch.Tensor) -> torch.Tensor:
                h = torch.tanh(states @ w1)
                out = torch.tanh(h @ w2)
                return out.squeeze(-1)

            return fn, "mlp"


class Encoder(nn.Module):
    """
    Transformer‑based encoder that maps a set of (state, reward)
    pairs to a latent distribution p(z | context).
    """

    def __init__(
        self,
        state_dim: int,
        latent_dim: int,
        n_layers: int = 4,
        n_heads: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.device = device

        # Project (state, reward) into transformer space
        # We concatenate reward as a scalar
        self.input_proj = nn.Linear(state_dim + 1, dim_feedforward)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_feedforward,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Latent mean & logvar
        self.fc_mu = nn.Linear(dim_feedforward, latent_dim)
        self.fc_logvar = nn.Linear(dim_feedforward, latent_dim)

    def forward(
        self, states: torch.Tensor, rewards: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        states: (B, K, state_dim)
        rewards: (B, K)
        """
        B, K, _ = states.shape
        # concatenate reward
        r_exp = rewards.unsqueeze(-1)  # (B, K, 1)
        x = torch.cat([states, r_exp], dim=-1)  # (B, K, state_dim+1)
        x = self.input_proj(x)  # (B, K, d_ff)
        x = self.transformer(x)  # (B, K, d_ff)
        x = x.mean(dim=1)  # (B, d_ff)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class Decoder(nn.Module):
    """
    MLP that predicts rewards given a state and latent code.
    """

    def __init__(
        self,
        state_dim: int,
        latent_dim: int,
        hidden_dims: List[int] = [512, 512, 512],
    ):
        super().__init__()
        layers = []
        dim_in = state_dim + latent_dim
        for h in hidden_dims:
            layers.append(nn.Linear(dim_in, h))
            layers.append(nn.ReLU())
            dim_in = h
        layers.append(nn.Linear(dim_in, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, states: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        states: (B, K, state_dim)
        z: (B, latent_dim) -> expand to (B, K, latent_dim)
        """
        B, K, _ = states.shape
        z_exp = z.unsqueeze(1).expand(-1, K, -1)
        inp = torch.cat([states, z_exp], dim=-1)
        out = self.net(inp)  # (B, K, 1)
        return out.squeeze(-1)  # (B, K)


class FRE(nn.Module):
    """
    Wrapper that contains the encoder and decoder.
    """

    def __init__(
        self,
        state_dim: int,
        latent_dim: int = 32,
        n_layers: int = 4,
        n_heads: int = 4,
        dim_feedforward: int = 256,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.encoder = Encoder(
            state_dim,
            latent_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            dim_feedforward=dim_feedforward,
            device=device,
        )
        self.decoder = Decoder(state_dim, latent_dim)

    def encode(
        self, states: torch.Tensor, rewards: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (z, mu, logvar)
        """
        mu, logvar = self.encoder(states, rewards)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar

    def decode(self, states: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(states, z)

    def forward(
        self,
        states_enc: torch.Tensor,
        rewards_enc: torch.Tensor,
        states_dec: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for training: encode context, decode on separate set.
        """
        z, mu, logvar = self.encode(states_enc, rewards_enc)
        preds = self.decode(states_dec, z)
        return preds, mu, logvar