import torch
import math
from src.utils import sinusoidal_embedding


class VESDE:
    """
    Variance‑exploding SDE for the toy Gaussian model.
    Parameters
    ----------
    sigma_min : float
        Minimal noise level (σ_0).
    sigma_max : float
        Maximal noise level (σ_T).
    """
    def __init__(self, sigma_min=0.01, sigma_max=10.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def sigma(self, t):
        """Return σ_t for t∈[0,1]"""
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t

    def sigma_sq(self, t):
        return self.sigma(t) ** 2

    def forward_transition(self, theta0, t):
        """
        Sample θ_t ∼ N(θ_0, σ_t² I)
        """
        sigma_t = self.sigma(t)
        eps = torch.randn_like(theta0)
        return theta0 + eps * sigma_t, sigma_t

    def score_noise_grad(self, theta_t, theta0, t):
        """
        Gradient of log p_{t|0}(θ_t | θ_0)
        For Gaussian noise: (θ_0 - θ_t) / σ_t²
        """
        sigma_sq = self.sigma_sq(t)
        return (theta0 - theta_t) / sigma_sq