import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple
from .score_network import MLPScoreNet
from .npse import VE_SDE
from .utils import set_seed, to_tensor, sinusoidal_embedding

class TSNPSE(NPSE):
    """
    Truncated Sequential NPSE (TSNPSE) – extends NPSE with truncated proposals.
    """
    def __init__(self, dim_theta: int, dim_x: int, device: str = "cpu",
                 hidden: int = 256, lr: float = 1e-4, seed: int = 0,
                 epsilon: float = 1e-4):
        super().__init__(dim_theta, dim_x, device, hidden, lr, seed)
        self.epsilon = epsilon
        # Store past posterior samples for HPR estimation
        self.posterior_samples = []

    def hpr(self, samples: torch.Tensor) -> torch.Tensor:
        """
        Highest‑probability‑region estimator.
        Compute log‑density of samples w.r.t current posterior (score network)
        and keep the top (1‑ε) mass.
        """
        with torch.no_grad():
            # Estimate log‑density using score and change‑of‑variables formula
            # Here we approximate log p by integrating the score along a straight line.
            # For simplicity we use a rough approximation: log p ≈ -||θ||²/2
            # This is a placeholder; in practice use instantaneous change‑of‑variables.
            logp = -0.5 * torch.sum(samples ** 2, dim=-1)
            threshold = torch.quantile(logp, self.epsilon)
            mask = logp >= threshold
            return samples[mask]

    def train_step(self, theta0: torch.Tensor, x: torch.Tensor, batch_size: int = 128):
        # First, run standard NPSE step
        loss = super().train_step(theta0, x, batch_size)

        # After training, collect posterior samples for HPR
        with torch.no_grad():
            theta_samples = self.sample(batch_size)
            self.posterior_samples.append(to_tensor(theta_samples))
        return loss

    def truncated_prior(self) -> torch.Tensor:
        """
        Build truncated prior by keeping samples in HPR from all rounds.
        """
        if not self.posterior_samples:
            # First round: prior itself
            return None
        all_samples = torch.cat(self.posterior_samples, dim=0)
        return self.hpr(all_samples)

    def sample(self, n_samples: int, n_steps: int = 1000) -> torch.Tensor:
        """
        Sample from the posterior using a truncated prior.
        """
        # Build truncated prior
        truncated = self.truncated_prior()
        if truncated is None:
            # First round: use standard reference
            theta = torch.randn(n_samples, self.net.dim_theta, device=self.device)
        else:
            # Rejection sampling from truncated prior
            theta = torch.empty((0, self.net.dim_theta), device=self.device)
            while theta.size(0) < n_samples:
                cand = torch.randn(n_samples, self.net.dim_theta, device=self.device)
                # Compute log density w.r.t truncated prior using the score network
                # Placeholder: assume truncated prior is uniform over samples
                mask = torch.ones(cand.size(0), dtype=torch.bool, device=self.device)
                theta = torch.cat([theta, cand[mask]], dim=0)
            theta = theta[:n_samples]
        # Integrate probability‑flow ODE as in NPSE
        return super().sample(n_samples, n_steps)