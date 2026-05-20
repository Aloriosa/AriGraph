import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt, log
from scipy.stats import norm, multivariate_normal


def sinusoidal_embedding(t, dim=64):
    """
    Sinusoidal embedding for a batch of scalar time values.

    Parameters
    ----------
    t : torch.Tensor of shape (B,)
        Scalar time values in the interval [0, 1].

    dim : int, optional (default=64)
        Dimension of the embedding.

    Returns
    -------
    torch.Tensor of shape (B, dim)
        Sinusoidal embeddings.
    """
    t = t.unsqueeze(-1)  # (B, 1)
    half = dim // 2
    positions = torch.arange(half, dtype=t.dtype, device=t.device)
    div_term = torch.exp(-torch.log(torch.tensor(10000.0, device=t.device)) * positions / half)
    emb = torch.zeros(t.size(0), dim, device=t.device)
    emb[:, :half] = torch.sin(t * div_term)
    emb[:, half:] = torch.cos(t * div_term)
    return emb


class MLP(nn.Module):
    """Simple 3‑layer MLP with SiLU activations."""
    def __init__(self, input_dim, output_dim, hidden=256, nlayers=3):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(nlayers):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.SiLU())
            in_dim = hidden
        layers.append(nn.Linear(hidden, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def estimate_gaussian_params(samples: torch.Tensor):
    """
    Estimate mean and covariance of a set of samples (B, d).

    Returns
    -------
    mean : torch.Tensor of shape (d,)
    cov : torch.Tensor of shape (d, d)
    """
    mean = torch.mean(samples, dim=0)
    diff = samples - mean
    cov = diff.T @ diff / (samples.size(0) - 1)
    return mean, cov


def posterior_density(x: torch.Tensor, mean: torch.Tensor, cov: torch.Tensor):
    """
    Evaluate the Gaussian density of points x under N(mean, cov).
    Works for both 1‑D and multi‑D samples.
    """
    if mean.numel() == 1:
        return torch.from_numpy(
            norm.pdf(x.cpu().numpy(), loc=mean.item(), scale=np.sqrt(cov.item()))
        ).to(x.device)
    else:
        mvn = multivariate_normal(mean=mean.cpu().numpy(), cov=cov.cpu().numpy())
        return torch.from_numpy(mvn.pdf(x.cpu().numpy())).to(x.device)