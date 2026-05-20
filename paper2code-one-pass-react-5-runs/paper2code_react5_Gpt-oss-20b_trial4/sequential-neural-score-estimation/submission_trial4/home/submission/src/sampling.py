import torch
import numpy as np
from .utils import sigma_t

def sample_posterior(model, x_obs, cfg, device, num_samples=5000, sample_steps=100):
    """
    Sample from the posterior using the probability‑flow ODE.
    """
    d = 1  # dimensionality of theta
    # Start from reference distribution p_T(θ_T) ≈ N(0, σ_T²)
    sigma_T = sigma_t(torch.tensor(cfg["T"], device=device),
                      cfg["sigma_min"], cfg["sigma_max"])
    theta = torch.randn(num_samples, d, device=device) * sigma_T

    t_vals = torch.linspace(cfg["T"], 0.0, steps=sample_steps, device=device)
    dt = t_vals[1] - t_vals[0]  # negative

    model.eval()
    with torch.no_grad():
        for t in t_vals[:-1]:
            # sigma at current time
            sigma = sigma_t(t, cfg["sigma_min"], cfg["sigma_max"])
            g = sigma
            # score at current theta
            score = model(theta, x_obs.expand_as(theta), t.expand(num_samples))
            # Probability‑flow ODE: dθ/dt = f - 0.5 g² * score (f = 0 for VE SDE)
            vel = -0.5 * g.pow(2) * score
            theta = theta + vel * dt  # Euler step

    return theta