import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

# ----------------------------------------------------------------
# Diffusion utilities
# ----------------------------------------------------------------
def ve_sde(t, sigma_min=0.01, sigma_max=50.0):
    """Variance‑exploding SDE:  dθ_t = σ_min (σ_min/σ_max)^t sqrt(2 log(σ_max/σ_min)) dW_t"""
    return sigma_min * (sigma_min / sigma_max) ** t

def ve_transition_mean(t, theta0):
    """Mean of the VE transition: E[θ_t | θ_0] = θ_0"""
    return theta0

def ve_transition_cov(t, sigma_min=0.01, sigma_max=50.0):
    """Covariance of the VE transition: Var(θ_t | θ_0) = σ_min^2 (σ_max/σ_min)^{2t} I"""
    return (sigma_min ** 2) * (sigma_max / sigma_min) ** (2 * t)

def sample_ve(theta0, t, sigma_min=0.01, sigma_max=50.0):
    """Sample θ_t from the VE transition given θ_0 and time t."""
    mean = ve_transition_mean(t, theta0)
    cov = ve_transition_cov(t, sigma_min, sigma_max)
    return np.random.multivariate_normal(mean, cov)

# ----------------------------------------------------------------
# Probability‑flow ODE integration (explicit RK4)
# ----------------------------------------------------------------
def probability_flow_ode(θ, t, score_fn, x, dt):
    """Compute one RK4 step of the probability‑flow ODE."""
    v = -ve_sde(t) ** 2 * 0.5 * score_fn(θ, x, t)  # f - ½ g² ∇log p_t
    k1 = v
    k2 = -ve_sde(t + dt/2) ** 2 * 0.5 * score_fn(θ + 0.5 * dt * k1, x, t + dt/2)
    k3 = -ve_sde(t + dt/2) ** 2 * 0.5 * score_fn(θ + 0.5 * dt * k2, x, t + dt/2)
    k4 = -ve_sde(t + dt)     ** 2 * 0.5 * score_fn(θ + dt * k3, x, t + dt)
    return θ + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def sample_from_ode(score_fn, x, n_steps=20, dt=0.05):
    """Sample from the approximate posterior by integrating the probability‑flow ODE."""
    # Start from a standard normal sample
    θ = np.random.randn(2)
    t = 1.0
    for _ in range(n_steps):
        θ = probability_flow_ode(θ, t, score_fn, x, dt)
        t -= dt
    return θ