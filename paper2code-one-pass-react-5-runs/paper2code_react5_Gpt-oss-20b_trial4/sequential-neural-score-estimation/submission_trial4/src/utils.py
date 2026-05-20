import torch
import numpy as np
from typing import Tuple

def sigma_t(t: torch.Tensor, sigma_min: float, sigma_max: float):
    """
    VE SDE scale factor sigma(t) = sigma_min * (sigma_max/sigma_min) ** t
    t is a tensor of shape (...,)
    """
    return sigma_min * (sigma_max / sigma_min) ** t

def drift_and_diffusion(theta_t: torch.Tensor, t: torch.Tensor, cfg):
    """
    For VE SDE: drift = 0, diffusion = sigma_t(t)
    Return (f, g)
    """
    f = torch.zeros_like(theta_t)
    g = sigma_t(t, cfg["sigma_min"], cfg["sigma_max"])
    return f, g

def target_score_with_likelihood(theta_t: torch.Tensor,
                                 theta_0: torch.Tensor,
                                 t: torch.Tensor,
                                 x_obs: torch.Tensor,
                                 cfg):
    """
    Target posterior score for the perturbed distribution:
    ∇_θ log p_{t|0}(θ_t | θ_0) = -(θ_t - θ_0) / σ_t^2
    plus the likelihood gradient ∇_θ log p(x_obs | θ_0):
        (x_obs - θ_0) / σ_sim^2
    plus the prior gradient ∇_θ log p(θ_0):
        -θ_0 / σ_prior^2
    """
    sigma = sigma_t(t, cfg["sigma_min"], cfg["sigma_max"])
    diff = -(theta_t - theta_0) / sigma.pow(2)
    # Likelihood gradient (Gaussian linear)
    sim_sigma = cfg["simulator_std"]
    lik_grad = (x_obs - theta_0) / (sim_sigma ** 2)
    # Prior gradient
    prior_sigma = cfg["prior_std"]
    prior_grad = -theta_0 / (prior_sigma ** 2)
    return diff + lik_grad + prior_grad

def log_posterior_gaussian(theta: torch.Tensor,
                           x_obs: float,
                           prior_std: float,
                           sim_std: float):
    """
    Log posterior density for the Gaussian‑linear benchmark:
    p(θ | x_obs) ∝ N(θ; μ_post, σ_post^2)
    where
        σ_post^2   = 1 / (1/σ_prior^2 + 1/σ_sim^2)
        μ_post     = σ_post^2 * (x_obs / σ_prior^2)
    The returned value is the log‑density up to an additive constant.
    """
    prior_var = prior_std ** 2
    sim_var   = sim_std ** 2
    var_post  = 1.0 / (1.0 / prior_var + 1.0 / sim_var)
    # Posterior mean (prior mean is 0 in this benchmark)
    mean_post = var_post * (x_obs / prior_var)
    # Log‑density (unnormalised)
    logp = -0.5 * ((theta - mean_post) ** 2) / var_post
    # Add constant term (optional, but useful for comparison)
    logp += -0.5 * torch.log(2 * torch.pi * var_post)
    return logp

def hpr_threshold(log_probs: torch.Tensor, epsilon: float):
    """
    Returns the log‑probability threshold such that the region with log‑probability >= threshold
    contains 1‑ε of the mass. We approximate by sorting.
    """
    sorted_logp, _ = torch.sort(log_probs, descending=True)
    idx = int(np.ceil((1.0 - epsilon) * len(sorted_logp)))
    # Clamp index to valid range
    idx = min(max(idx, 1), len(sorted_logp))
    return sorted_logp[idx - 1]  # -1 because idx is 1‑indexed