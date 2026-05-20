"""
Implementation of the core Simformer components:
  * a simple transformer that takes the joint vector [θ, x] as a single token
  * a diffusion module (VESDE) that provides noise schedules
  * helper functions for forward and reverse diffusion
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleTransformerScore(nn.Module):
    """
    A minimal transformer that learns the score of the joint distribution.
    Input: (batch, dim)  where dim = 4 (θ₁, θ₂, x₁, x₂)
    Output: (batch, dim) – predicted score vector.
    """
    def __init__(self, dim: int = 4, nhead: int = 4, num_layers: int = 4, dim_feedforward: int = 128):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, dim)
        Returns: (batch, dim) – predicted score
        """
        # The transformer expects (batch, seq_len, dim). We use seq_len=1.
        out = self.encoder(x.unsqueeze(1)).squeeze(1)
        return out


class VESDE:
    """
    Variance‑exploding SDE as defined in Song & Ermon 2021b.
    """
    def __init__(self, sigma_min: float = 0.01, sigma_max: float = 15.0, beta_min: float = 0.01, beta_max: float = 10.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.beta_min = beta_min
        self.beta_max = beta_max

    def sigma(self, t: float) -> float:
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t * math.sqrt(2 * math.log(self.sigma_max / self.sigma_min))

    def noise(self, t: float, shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
        """
        Return noise ε ∼ N(0, I) scaled by σ(t)
        """
        eps = torch.randn(shape, device=device)
        return eps * self.sigma(t)

    def forward_sample(self, x0: torch.Tensor, t: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample x_t from the forward process: x_t = x0 + σ(t) * ε
        Returns (x_t, ε)
        """
        eps = torch.randn_like(x0)
        sigma_t = self.sigma(t)
        xt = x0 + sigma_t * eps
        return xt, eps

    def reverse_score(self, model: nn.Module, xt: torch.Tensor, t: float) -> torch.Tensor:
        """
        Compute the drift term needed for the reverse SDE:
        f_rev = -g(t)^2 * s(x_t, t)
        For VESDE, f_rev = -σ(t)^2 * s
        """
        sigma_t = self.sigma(t)
        s = model(xt)  # predicted score
        return -sigma_t ** 2 * s

    def reverse_step(self, model: nn.Module, xt: torch.Tensor, t: float, dt: float) -> torch.Tensor:
        """
        One reverse SDE step using Euler–Maruyama.
        """
        sigma_t = self.sigma(t)
        s = model(xt)
        drift = -sigma_t ** 2 * s
        noise = torch.randn_like(xt) * sigma_t * math.sqrt(-dt)
        return xt + drift * dt + noise