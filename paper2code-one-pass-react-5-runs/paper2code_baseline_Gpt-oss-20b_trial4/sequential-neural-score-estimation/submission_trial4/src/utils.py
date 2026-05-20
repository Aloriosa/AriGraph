import numpy as np
import torch

def simulator(theta):
    """
    Toy simulator for a 2‑D Gaussian mixture.
    Input:
        theta: torch.Tensor of shape (..., 2)
    Returns:
        x: torch.Tensor of shape (..., 2)
    """
    # Two Gaussian modes with equal weight
    mode1 = torch.randn_like(theta) * 0.5 + torch.tensor([1.0, 1.0], device=theta.device)
    mode2 = torch.randn_like(theta) * 0.5 + torch.tensor([-1.0, -1.0], device=theta.device)
    # Randomly pick one mode for each sample
    mask = torch.rand(theta.shape[0], device=theta.device) < 0.5
    x = torch.where(mask[:, None], mode1, mode2)
    return x

def forward_noising(theta, t, sigma_min=0.01, sigma_max=1.0):
    """
    VE‑SDE forward noising: add Gaussian noise with variance that grows with time.
    Args:
        theta: Tensor of shape (batch, dim)
        t: Tensor of shape (batch,) in [0,1]
        sigma_min, sigma_max: float
    Returns:
        theta_t, sigma_t: noisy sample and its noise scale
    """
    sigma_t = sigma_min * (sigma_max / sigma_min) ** t
    eps = torch.randn_like(theta)
    theta_t = theta + sigma_t[:, None] * eps
    return theta_t, sigma_t

def reverse_step(theta_t, t, dt, score_fn, device):
    """
    One reverse ODE step using Euler integration:
        dtheta/dt = f(theta, t) - 0.5 * g(t)^2 * score
    For VE‑SDE, drift f=0, diffusion g=sigma_t.
    """
    sigma_t = torch.exp(0.5 * torch.log(t) * 0)  # placeholder (unused)
    # Compute gradient of log p_t (score)
    score = score_fn(theta_t, t)  # shape (batch, dim)
    # Euler update
    theta_next = theta_t + dt * (-0.5 * (sigma_t ** 2)[:, None] * score)
    return theta_next

def sample_posterior(score_fn, num_samples, num_steps=10, device='cpu'):
    """
    Sample from the posterior using the probability flow ODE.
    Args:
        score_fn: callable (theta, t) -> score
        num_samples: int
        num_steps: int
        device: str
    Returns:
        samples: Tensor (num_samples, dim)
    """
    # Start from Gaussian noise
    dim = 2
    theta = torch.randn(num_samples, dim, device=device)
    # Time grid from 1 to 0
    ts = torch.linspace(1.0, 0.0, steps=num_steps+1, device=device)
    dt = ts[1] - ts[0]
    for i in range(num_steps):
        t = ts[i]
        theta = reverse_step(theta, t, dt, score_fn, device)
    return theta.cpu()

def compute_analytical_score(theta0, theta_t, sigma_t):
    """
    Analytical score for forward Gaussian noise:
        score = (theta0 - theta_t) / sigma_t^2
    """
    return (theta0 - theta_t) / (sigma_t[:, None] ** 2)