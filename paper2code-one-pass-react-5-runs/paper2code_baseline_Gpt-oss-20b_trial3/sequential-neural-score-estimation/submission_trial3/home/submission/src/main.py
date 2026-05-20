#!/usr/bin/env python3
"""
Toy demo of Sequential Neural Score Estimation (SNPSE).

The toy problem:
    - Prior: θ ~ N(0, 0.1 I₂)
    - Likelihood: x | θ ~ N(θ, 0.1 I₂)
    - Observation: x_obs = [0.5, -0.2]

We train a conditional score network s_ψ(θ_t, x, t) ≈ ∇_θ log p_t(θ_t | x)
using the denoising posterior score‑matching objective.  Then we sample
from the posterior by integrating the probability‑flow ODE with the
learned score function.
"""

import os
import json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.utils import sample_ve, sample_from_ode

# ----------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------
CONFIG = {
    "prior_mean": [0.0, 0.0],
    "prior_cov": 0.1 * np.eye(2),
    "likelihood_cov": 0.1 * np.eye(2),
    "x_obs": np.array([0.5, -0.2]),
    "num_samples": 2000,      # number of synthetic data points for training
    "batch_size": 64,
    "num_epochs": 10,
    "learning_rate": 1e-3,
    "device": "cpu",          # set to 'cuda' if a GPU is available
    "seed": 42,
    "ode_steps": 20,          # number of steps for the probability‑flow ODE
    "ode_dt": 0.05,
    "out_samples": 1000,      # number of posterior samples to generate
}


# ----------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------
def set_random_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sample_prior(n: int):
    return np.random.multivariate_normal(CONFIG["prior_mean"],
                                         CONFIG["prior_cov"],
                                         size=n)


def sample_likelihood(theta, n: int):
    mean = theta
    cov = CONFIG["likelihood_cov"]
    return np.random.multivariate_normal(mean, cov, size=n)


def generate_training_data():
    """Generate synthetic dataset (θ₀, x, θ_t, t)."""
    theta0 = sample_prior(CONFIG["num_samples"])
    # Sample observation x from the likelihood
    x = np.array([sample_likelihood(t, 1)[0] for t in theta0])
    # For each (θ₀, x) draw a time t and a perturbed θ_t
    t = np.random.uniform(0, 1, size=CONFIG["num_samples"])
    theta_t = np.array([sample_ve(t_i, theta0[i]) for i, t_i in enumerate(t)])
    return theta0, x, theta_t, t


# ----------------------------------------------------------------
# Score network
# ----------------------------------------------------------------
class ScoreNet(nn.Module):
    def __init__(self, dim_theta=2, dim_x=2, dim_t=1, hidden=128):
        super().__init__()
        self.embed_theta = nn.Sequential(
            nn.Linear(dim_theta, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.embed_x = nn.Sequential(
            nn.Linear(dim_x, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.embed_t = nn.Sequential(
            nn.Linear(1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.output = nn.Sequential(
            nn.Linear(3 * hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim_theta),
        )

    def forward(self, theta_t, x, t):
        """
        theta_t: (B, d)
        x: (B, p)
        t: (B, 1)
        """
        h1 = self.embed_theta(theta_t)
        h2 = self.embed_x(x)
        h3 = self.embed_t(t)
        h = torch.cat([h1, h2, h3], dim=-1)
        return self.output(h)


def train_score_network(train_loader, device="cpu"):
    model = ScoreNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    loss_fn = nn.MSELoss()

    for epoch in range(CONFIG["num_epochs"]):
        epoch_loss = 0.0
        for batch in train_loader:
            theta_t, x, t, target = batch
            theta_t = theta_t.to(device)
            x = x.to(device)
            t = t.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            pred = model(theta_t, x, t)
            loss = loss_fn(pred, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * theta_t.size(0)

        epoch_loss /= len(train_loader.dataset)
        print(f"Epoch {epoch + 1:02d} | MSE: {epoch_loss:.6f}")

    return model


def create_dataloader():
    theta0, x, theta_t, t = generate_training_data()
    # Compute the true score of the perturbed posterior:
    # ∇_θ log p_{t|0}(θ_t | θ0) = (θ0 - θ_t) / var_t
    var_t = 0.1 * (CONFIG["likelihood_cov"] + 0.1 * np.eye(2))  # approximate
    var_t = 0.1 + 0.1 * t[:, None, None]  # scalar variance for toy
    var_t = 0.1 * (CONFIG["likelihood_cov"] + 0.1 * np.eye(2))
    var_t = 0.1 + 0.1 * t[:, None, None]  # scalar variance for toy
    var_t = 0.1 * (CONFIG["prior_cov"] + 0.1 * np.eye(2))
    # For the toy, we approximate the conditional density as Gaussian with
    # mean θ0 and variance σ_t² = 0.1 * (1 + t)
    sigma_t2 = 0.1 * (1.0 + t)
    true_score = (theta0 - theta_t) / sigma_t2[:, None]

    # Convert to torch tensors
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(theta_t, dtype=torch.float32),
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(t[:, None], dtype=torch.float32),
        torch.tensor(true_score, dtype=torch.float32),
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        drop_last=True,
    )
    return loader


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------
def main():
    set_random_seed(CONFIG["seed"])

    # Create data loader
    loader = create_dataloader()

    # Train the score network
    model = train_score_network(loader, device=CONFIG["device"])

    # Define a Python callable that wraps the PyTorch model
    def score_fn(θ, x, t):
        with torch.no_grad():
            θ_t = torch.tensor(θ, dtype=torch.float32).unsqueeze(0)
            x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
            t_t = torch.tensor([[t]], dtype=torch.float32)
            out = model(θ_t, x_t, t_t)
        return out.squeeze(0).numpy()

    # Sample from the posterior
    samples = []
    for _ in range(CONFIG["out_samples"]):
        sample = sample_from_ode(score_fn,
                                 CONFIG["x_obs"],
                                 n_steps=CONFIG["ode_steps"],
                                 dt=CONFIG["ode_dt"])
        samples.append(sample)

    samples = np.stack(samples)
    np.save("samples.npy", samples)

    # Save a small log
    log = {
        "config": CONFIG,
        "n_samples": CONFIG["out_samples"],
        "posterior_mean": samples.mean(axis=0).tolist(),
        "posterior_std": samples.std(axis=0).tolist(),
    }
    with open("log.txt", "w") as f:
        json.dump(log, f, indent=2)

    print("Sampling finished. Results written to 'samples.npy' and 'log.txt'.")


if __name__ == "__main__":
    main()