"""
Diffusion utilities – VESDE definition, time embeddings, and reverse SDE solver.
"""

import torch
import math
import numpy as np
from torch import nn


class VESDE(nn.Module):
    """
    Variance‑Exploding SDE.
    dX_t = 0 * dt + g(t) dW_t
    with g(t) = sigma_min * (sigma_max / sigma_min) ** t
    """

    def __init__(self, sigma_min=1e-2, sigma_max=20.0, beta_min=0.01, beta_max=10.0):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.beta_min = beta_min
        self.beta_max = beta_max

    def sigma(self, t):
        """Noise scale at time t ∈ [0, 1]"""
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t

    def mean_coeff(self, t):
        """μ_t(x0) = x0 for VESDE"""
        return 1.0

    def std_coeff(self, t):
        """σ_t = sigma(t)"""
        return self.sigma(t)

    def score(self, x_t, x0, t):
        """Analytic score for VESDE:
        s(x_t, t) = (x_t - μ_t(x0)) / σ_t^2
        """
        sigma_t = self.std_coeff(t)
        return (x_t - x0) / (sigma_t ** 2)

    def reverse_step(self, x_t, s, t, dt, eps):
        """
        Euler–Maruyama reverse step:
        dX_t = -g(t)^2 * s * dt + g(t) dW_t
        """
        sigma_t = self.std_coeff(t)
        return x_t + (- (sigma_t ** 2) * s) * dt + sigma_t * math.sqrt(dt) * eps


class TimeEmbedding(nn.Module):
    """
    Random Fourier embedding of diffusion time t ∈ [0,1].
    """

    def __init__(self, embed_dim: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self.register_buffer("omega", torch.randn(embed_dim, dtype=torch.float32))
        self.register_buffer("b", torch.rand(embed_dim, dtype=torch.float32) * 2 * math.pi)

    def forward(self, t):
        # t: [batch] or scalar
        t = t.unsqueeze(-1) if isinstance(t, torch.Tensor) else torch.tensor(t).unsqueeze(-1)
        return torch.cos(t * self.omega + self.b)