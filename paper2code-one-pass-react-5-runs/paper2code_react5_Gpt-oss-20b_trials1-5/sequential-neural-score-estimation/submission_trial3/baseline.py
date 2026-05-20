#!/usr/bin/env python3
"""
Simple SNPE baseline implementation.

The baseline trains a conditional density estimator that predicts a
Gaussian posterior p(θ | x).  The model is an MLP that outputs
the mean and log‑variance of the posterior diagonal covariance.

Author: OpenAI ChatGPT
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np


class SNPEModel(nn.Module):
    """
    Conditional density estimator for SNPE.
    Outputs mean and log‑variance of a diagonal Gaussian posterior.
    """

    def __init__(self, dim_theta: int, dim_x: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_x, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim_theta * 2),  # mean + logvar
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        mean, logvar = out.chunk(2, dim=-1)
        return mean, logvar


def train_snpe(
    model: nn.Module,
    dataset: list,
    device: torch.device,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-4,
    verbose: bool = True,
) -> None:
    """
    Train the SNPE model on the supplied dataset.
    Each element in `dataset` is a tuple (theta0, x) of torch.Tensors.
    """
    all_theta = torch.cat([t for t, _ in dataset], dim=0).to(device)
    all_x = torch.cat([x for _, x in dataset], dim=0).to(device)

    ds = TensorDataset(all_x, all_theta)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for xb, tb in loader:
            xb, tb = xb.to(device), tb.to(device)
            mean, logvar = model(xb)
            var = torch.exp(logvar)
            # Gaussian negative log‑likelihood
            nll = 0.5 * ((tb - mean) ** 2 / var + logvar + np.log(2 * np.pi))
            loss = nll.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if verbose and (epoch + 1) % 5 == 0:
            print(f"  SNPE Epoch {epoch+1}/{epochs} – loss: {epoch_loss/len(loader):.4f}")


def sample_snpe(
    model: nn.Module,
    x_obs: torch.Tensor,
    device: torch.device,
    n_samples: int = 5000,
) -> np.ndarray:
    """
    Draw samples from the posterior predicted by the SNPE model.
    """
    model.eval()
    x_obs = x_obs.expand(n_samples, -1).to(device)
    with torch.no_grad():
        mean, logvar = model(x_obs)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mean)
        samples = mean + eps * std
    return samples.cpu().numpy()