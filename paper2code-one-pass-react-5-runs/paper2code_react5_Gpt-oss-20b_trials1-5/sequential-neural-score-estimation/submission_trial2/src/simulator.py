"""
Toy simulator for a 1‑D Gaussian model.
Prior: θ ~ N(0, 1)
Likelihood: x | θ ~ N(θ, 0.1)
"""

import torch
import math

def sample_prior(num):
    """Sample n parameters from the prior p(θ)=N(0,1)."""
    return torch.randn(num, 1)

def sample_reference(num, sigma_T):
    """Sample n parameters from the reference distribution π = N(0, σ_T²)."""
    return torch.randn(num, 1) * sigma_T

def simulate(theta):
    """
    Simulate a single data point for each θ.
    θ : tensor of shape (n, 1)
    Returns x of shape (n, 1)
    """
    noise = torch.randn_like(theta) * math.sqrt(0.1)
    return theta + noise