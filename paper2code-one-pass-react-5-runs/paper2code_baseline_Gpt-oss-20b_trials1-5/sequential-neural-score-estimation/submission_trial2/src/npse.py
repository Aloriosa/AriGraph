import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple
from .score_network import MLPScoreNet
from .utils import set_seed, to_tensor

class VE_SDE:
    """
    Variance‑exploding SDE used for perturbation.
    Transition: θ_t | θ_0 ~ N(θ_0, σ_t² I)
    σ_t = σ_min * (σ_max/σ_min)^t
    """
    def __init__(self, sigma_min: float = 0.01, sigma_max: float = 10.0, device: str = "cpu"):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.device = device

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t

    def sample(self, theta0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Sample θ_t from θ_0 using the SDE transition."""
        sigma_t = self.sigma(t)
        eps = torch.randn_like(theta0)
        return theta0 + sigma_t.unsqueeze(-1) * eps

    def target_score(self, theta_t: torch.Tensor, theta0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        ∇_{θ_t} log p_{t|0}(θ_t | θ_0) = (θ_0 - θ_t) / σ_t²
        """
        sigma_t = self.sigma(t)
        return (theta0 - theta_t) / (sigma_t**2).unsqueeze(-1)

class NPSE:
    """
    Non‑Sequential Neural Posterior Score Estimation.
    """
    def __init__(self, dim_theta: int, dim_x: int, device: str = "cpu",
                 hidden: int = 256, lr: float = 1e-4, seed: int = 0):
        set_seed(seed)
        self.device = device
        self.net = MLPScoreNet(dim_theta, dim_x, hidden).to(device)
        self.sde = VE_SDE(device=device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

    def train_step(self, theta0: torch.Tensor, x: torch.Tensor, batch_size: int = 128):
        self.net.train()
        B, d = theta0.shape
        # Random times
        t = torch.rand(batch_size, device=self.device)
        # Sample θ_t
        theta_t = self.sde.sample(theta0[:batch_size], t)
        # Target score
        target = self.sde.target_score(theta_t, theta0[:batch_size], t)
        # Predicted score
        pred = self.net(theta_t, x[:batch_size], t)
        loss = 0.5 * torch.mean((pred - target) ** 2)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sample(self, n_samples: int, n_steps: int = 1000) -> torch.Tensor:
        """
        Sample from the posterior using the probability‑flow ODE.
        Integrate from t=1 → 0 using simple Euler.
        """
        self.net.eval()
        device = self.device
        # Start from reference: θ_1 ~ N(0, I)
        theta = torch.randn(n_samples, self.net.dim_theta, device=device)
        t = torch.ones(n_samples, device=device)
        dt = -1.0 / n_steps
        for _ in range(n_steps):
            # compute velocity: v = -f + 0.5 g² ∇_θ log p_t(θ_t | x)
            # Here f = 0, g = σ_t, so v = 0.5 σ_t² * pred
            pred = self.net(theta, torch.ones(n_samples, self.net.dim_x, device=device), t)
            sigma_t = self.sde.sigma(t)
            v = 0.5 * (sigma_t**2).unsqueeze(-1) * pred
            theta = theta + v * dt
            t = t + dt
        return theta.detach().cpu().numpy()

    def save(self, path: str):
        torch.save(self.net.state_dict(), path)

    def load(self, path: str):
        self.net.load_state_dict(torch.load(path, map_location=self.device))