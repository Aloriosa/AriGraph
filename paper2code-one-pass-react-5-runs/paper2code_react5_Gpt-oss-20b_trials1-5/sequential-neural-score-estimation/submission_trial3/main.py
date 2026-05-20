#!/usr/bin/env python3
"""
Sequential Neural Posterior Score Estimation (TSNPSE) – Toy Benchmark Implementation.
Also includes a simple SNPE baseline for comparison.

Author:  OpenAI ChatGPT
Date:    2026-03-16
"""

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.neighbors import KernelDensity
from torch.utils.data import DataLoader, TensorDataset

# Import baseline
from baseline import SNPEModel, train_snpe, sample_snpe

# --------------------------------------------------------------------------- #
# 1. Utility functions
# --------------------------------------------------------------------------- #
def set_seed(seed: int = 42):
    """Set all relevant random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_sigma(t: torch.Tensor, sigma_min: float = 0.01, sigma_max: float = 10.0) -> torch.Tensor:
    """Diffusion coefficient for the variance‑exploding SDE."""
    return sigma_min * torch.exp(t * math.log(sigma_max / sigma_min))


def sinusoidal_embedding(t: torch.Tensor, num_bands: int = 10) -> torch.Tensor:
    """Sinusoidal time embedding."""
    device = t.device
    scales = 2.0 ** torch.arange(num_bands, device=device)
    emb = torch.cat([torch.sin(t * scales), torch.cos(t * scales)], dim=-1)
    return emb


# --------------------------------------------------------------------------- #
# 2. Score network
# --------------------------------------------------------------------------- #
class ScoreNetwork(nn.Module):
    """
    Conditional score network s_ψ(θ_t, x, t) → ∇_θ log p_t(θ_t | x).
    Architecture: 3‑layer MLPs for θ, x, t embeddings + 3‑layer MLP for output.
    """

    def __init__(self, dim_theta: int, dim_x: int, hidden_dim: int = 256, time_emb_dim: int = 20):
        super().__init__()
        # Theta embedding
        self.theta_emb = nn.Sequential(
            nn.Linear(dim_theta, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        # X embedding
        self.x_emb = nn.Sequential(
            nn.Linear(dim_x, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        # Time embedding
        self.t_emb = nn.Sequential(
            nn.Linear(2 * time_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        # Final network
        self.net = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim_theta),  # output dimension = dim_theta
        )

    def forward(self, theta: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        theta: (B, d)
        x:     (B, p)
        t:     (B, 1)
        """
        theta_e = self.theta_emb(theta)
        x_e = self.x_emb(x)
        t_e = self.t_emb(sinusoidal_embedding(t))
        h = torch.cat([theta_e, x_e, t_e], dim=-1)
        return self.net(h)


# --------------------------------------------------------------------------- #
# 3. Benchmarks
# --------------------------------------------------------------------------- #
def gaussian_linear_prior_sampler(batch_size: int, device: torch.device) -> torch.Tensor:
    # θ ~ N(0, 0.1 I)
    return torch.randn(batch_size, 2, device=device) * math.sqrt(0.1)


def gaussian_mixture_prior_sampler(batch_size: int, device: torch.device) -> torch.Tensor:
    # θ ~ Uniform(-10, 10)
    return torch.empty(batch_size, 2, device=device).uniform_(-10, 10)


def two_moons_prior_sampler(batch_size: int, device: torch.device) -> torch.Tensor:
    # θ ~ Uniform(-1, 1)^2
    return torch.empty(batch_size, 2, device=device).uniform_(-1, 1)


def gaussian_linear_simulator(batch_size: int, device: torch.device,
                              theta0: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prior: θ ~ N(0, 0.1 I). Likelihood: x | θ ~ N(θ, 0.1 I)."""
    if theta0 is None:
        theta0 = gaussian_linear_prior_sampler(batch_size, device)
    noise = torch.randn(batch_size, 2, device=device) * math.sqrt(0.1)
    x = theta0 + noise
    return theta0, x


def gaussian_mixture_simulator(batch_size: int, device: torch.device,
                               theta0: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prior: θ ~ Uniform(-10, 10). Likelihood: mixture."""
    if theta0 is None:
        theta0 = gaussian_mixture_prior_sampler(batch_size, device)
    comp = torch.bernoulli(torch.full((batch_size, 1), 0.5, device=device)).squeeze(-1)
    std = torch.where(comp == 1, torch.ones_like(comp), torch.full_like(comp, 0.1))
    noise = torch.randn(batch_size, 2, device=device) * std.unsqueeze(-1)
    x = theta0 + noise
    return theta0, x


def two_moons_simulator(batch_size: int, device: torch.device,
                        theta0: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prior: θ ~ Uniform(-1, 1)^2. Simulator as described in the paper."""
    if theta0 is None:
        theta0 = two_moons_prior_sampler(batch_size, device)
    alpha = torch.empty(batch_size, 1, device=device).uniform_(-math.pi / 2, math.pi / 2)
    r = torch.randn(batch_size, 1, device=device) * 0.01 + 0.1
    moon = torch.cat([r * torch.cos(alpha), r * torch.sin(alpha)], dim=-1)
    shift = torch.cat(
        [
            -torch.abs(theta0[:, 0] + theta0[:, 1]) / math.sqrt(2),
            (-theta0[:, 0] + theta0[:, 1]) / math.sqrt(2),
        ],
        dim=-1,
    )
    x = moon + shift
    return theta0, x


BENCHMARKS: List[Dict] = [
    {
        "name": "gaussian_linear",
        "dim_theta": 2,
        "dim_x": 2,
        "sim_func": gaussian_linear_simulator,
        "prior_sampler": gaussian_linear_prior_sampler,
        "x_obs": torch.tensor([1.0, -1.0]),
    },
    {
        "name": "gaussian_mixture",
        "dim_theta": 2,
        "dim_x": 2,
        "sim_func": gaussian_mixture_simulator,
        "prior_sampler": gaussian_mixture_prior_sampler,
        "x_obs": torch.tensor([0.5, -0.5]),
    },
    {
        "name": "two_moons",
        "dim_theta": 2,
        "dim_x": 2,
        "sim_func": two_moons_simulator,
        "prior_sampler": two_moons_prior_sampler,
        "x_obs": torch.tensor([0.2, 0.3]),
    },
]


# --------------------------------------------------------------------------- #
# 4. Truncated prior sampler
# --------------------------------------------------------------------------- #
def truncated_prior_sampler(
    n: int,
    prior_sampler: Callable[[int, torch.device], torch.Tensor],
    kde: KernelDensity,
    threshold: float,
    device: torch.device,
    max_attempts: int = 1000,
) -> torch.Tensor:
    """
    Sample n θ from the prior, accepting only those whose KDE density
    under the previous posterior exceeds the given threshold.
    """
    samples = []
    batch_size = max(2 * n, 1000)
    attempts = 0
    while len(samples) < n and attempts < max_attempts:
        cand = prior_sampler(batch_size, device)
        dens = np.exp(kde.score_samples(cand.cpu().numpy()))
        accepted = cand[dens > threshold]
        if accepted.numel() > 0:
            samples.append(accepted)
        attempts += 1
    if len(samples) == 0:
        raise RuntimeError("Could not sample from truncated prior (max attempts reached)")
    return torch.cat(samples, dim=0)[:n]


# --------------------------------------------------------------------------- #
# 5. Training loop
# --------------------------------------------------------------------------- #
def train_model_on_dataset(
    model: nn.Module,
    dataset: List[Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    lambda_t: Callable[[torch.Tensor], torch.Tensor] = lambda t: torch.ones_like(t),
    sigma_min: float = 0.01,
    sigma_max: float = 10.0,
    val_frac: float = 0.15,
) -> None:
    """
    Train the score network on all samples in `dataset` for `epochs` epochs.
    Uses a validation split to keep the best model.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Prepare data
    all_theta0 = torch.cat([t for t, _ in dataset], dim=0)
    all_x = torch.cat([x for _, x in dataset], dim=0)
    num_samples = all_theta0.size(0)
    # Shuffle and split
    indices = torch.randperm(num_samples)
    split = int(num_samples * (1 - val_frac))
    train_idx, val_idx = indices[:split], indices[split:]
    train_ds = TensorDataset(all_theta0[train_idx], all_x[train_idx])
    val_ds = TensorDataset(all_theta0[val_idx], all_x[val_idx])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_theta0, batch_x in train_loader:
            batch_theta0 = batch_theta0.to(device)
            batch_x = batch_x.to(device)
            t = torch.rand(batch_theta0.size(0), 1, device=device)
            sigma_t = get_sigma(t, sigma_min, sigma_max)
            eps = torch.randn_like(batch_theta0) * sigma_t
            theta_t = batch_theta0 + eps
            # Target score for conditional denoising posterior score matching
            target_score = (theta_t - batch_theta0) / sigma_t**2
            pred_score = model(theta_t, batch_x, t)
            loss = ((pred_score - target_score) ** 2).mean() * lambda_t(t).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_epoch_loss = epoch_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_theta0, batch_x in val_loader:
                batch_theta0 = batch_theta0.to(device)
                batch_x = batch_x.to(device)
                t = torch.rand(batch_theta0.size(0), 1, device=device)
                sigma_t = get_sigma(t, sigma_min, sigma_max)
                eps = torch.randn_like(batch_theta0) * sigma_t
                theta_t = batch_theta0 + eps
                target_score = (theta_t - batch_theta0) / sigma_t**2
                pred_score = model(theta_t, batch_x, t)
                loss = ((pred_score - target_score) ** 2).mean() * lambda_t(t).mean()
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)

        # Checkpoint best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs} – train loss: {avg_epoch_loss:.4f} | val loss: {avg_val_loss:.4f}")

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)


# --------------------------------------------------------------------------- #
# 6. Posterior sampling
# --------------------------------------------------------------------------- #
def sample_posterior(
    model: nn.Module,
    x_obs: torch.Tensor,
    device: torch.device,
    num_samples: int = 5000,
    dt: float = 0.01,
    sigma_min: float = 0.01,
    sigma_max: float = 10.0,
) -> np.ndarray:
    """
    Sample from the posterior using the probability‑flow ODE backwards in time.
    """
    model.eval()
    samples = []

    dim_theta = x_obs.size(0)
    for _ in tqdm(range(num_samples), desc="Sampling posterior"):
        # Start from reference distribution at t=1: N(0, sigma_max^2 I)
        theta = torch.randn(1, dim_theta, device=device) * sigma_max
        t = torch.ones(1, 1, device=device)
        while t.item() > 0.0:
            sigma_t = get_sigma(t, sigma_min, sigma_max)
            sigma_t_sq = sigma_t**2
            with torch.no_grad():
                score = model(theta, x_obs.expand_as(theta), t)
                velocity = 0.5 * sigma_t_sq * score
                theta = theta - velocity * dt
            t = t - dt
            if t.item() < 0.0:
                t = torch.zeros_like(t)
        samples.append(theta.cpu().numpy().flatten())

    return np.vstack(samples)


# --------------------------------------------------------------------------- #
# 7. Analytic posterior for Gaussian linear
# --------------------------------------------------------------------------- #
def analytic_posterior_gaussian_linear(x_obs: torch.Tensor, var_prior=0.1, var_lik=0.1):
    """Return mean and covariance of the analytic posterior."""
    var_post = 1.0 / (1.0 / var_prior + 1.0 / var_lik)
    mean_post = var_post * (x_obs / var_lik)
    cov_post = var_post * torch.eye(len(x_obs))
    return mean_post, cov_post


# --------------------------------------------------------------------------- #
# 8. KL divergence via KDE
# --------------------------------------------------------------------------- #
def kl_divergence_from_kde(samples: np.ndarray, kde_true: KernelDensity, kde_est: KernelDensity) -> float:
    """
    Estimate KL divergence KL(p_true || p_est) using samples from the true density.
    """
    # Evaluate log densities
    log_p_true = kde_true.score_samples(samples)
    log_p_est = kde_est.score_samples(samples)
    return np.mean(log_p_true - log_p_est)


# --------------------------------------------------------------------------- #
# 9. Effective sample size (ES)
# --------------------------------------------------------------------------- #
def effective_sample_size(weights: np.ndarray) -> float:
    """
    Compute effective sample size from a weight array.
    For equal weights, ES = len(weights).  This function is kept for completeness.
    """
    weights = weights / np.sum(weights)
    return 1.0 / np.sum(weights**2)


# --------------------------------------------------------------------------- #
# 10. Coverage computation (per‑dimension)
# --------------------------------------------------------------------------- #
def coverage_per_dim(samples: np.ndarray, true_theta: np.ndarray,
                     levels: List[float] = [0.5, 0.8, 0.9, 0.95]) -> Dict[float, float]:
    """
    Compute per‑dimension coverage proportions and return the average over dimensions.
    """
    cov_dict = {}
    for level in levels:
        lower = np.quantile(samples, (1 - level) / 2.0, axis=0)
        upper = np.quantile(samples, 1 - (1 - level) / 2.0, axis=0)
        inside = (true_theta >= lower) & (true_theta <= upper)
        cov = np.mean(inside)
        cov_dict[level] = float(cov)
    return cov_dict


# --------------------------------------------------------------------------- #
# 11. Main experiment loop
# --------------------------------------------------------------------------- #
def run_experiment(
    benchmark: Dict,
    output_dir: Path,
    device: torch.device,
    rounds: int = 3,
    sims_per_round: int = 2000,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-4,
    epsilon: float = 0.05,
    sigma_min: float = 0.01,
    sigma_max: float = 10.0,
    baseline_epochs: int = 10,
):
    name = benchmark["name"]
    dim_theta = benchmark["dim_theta"]
    dim_x = benchmark["dim_x"]
    sim_func = benchmark["sim_func"]
    prior_sampler = benchmark["prior_sampler"]
    x_obs = benchmark["x_obs"].to(device)

    print(f"\n=== Benchmark: {name} ===")
    out_path = output_dir / name
    out_path.mkdir(parents=True, exist_ok=True)

    # Create a single model instance that will be updated each round
    model = ScoreNetwork(dim_theta, dim_x).to(device)
    # Baseline SNPE model
    snpe_model = SNPEModel(dim_theta, dim_x).to(device)

    dataset: List[Tuple[torch.Tensor, torch.Tensor]] = []
    kde: KernelDensity | None = None
    threshold: float | None = None

    for rnd in range(1, rounds + 1):
        print(f"--- Round {rnd}/{rounds} ---")

        # 1. Generate new simulation batch
        if rnd == 1:
            theta0, x = sim_func(sims_per_round, device)
        else:
            # Sample from truncated prior using KDE from previous round
            theta0 = truncated_prior_sampler(
                sims_per_round,
                prior_sampler,
                kde,
                threshold,
                device,
            )
            theta0, x = sim_func(sims_per_round, device, theta0)
        dataset.append((theta0, x))
        print(f"  Round {rnd} data: {theta0.size(0)} samples")

        # 2. Train TSNPSE on accumulated data
        print("  Training TSNPSE model...")
        train_model_on_dataset(
            model,
            dataset,
            device,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )

        # 3. After training, sample from posterior to fit KDE for next round
        print("  Sampling posterior for KDE...")
        posterior_samples = sample_posterior(
            model,
            x_obs,
            device,
            num_samples=5000,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )
        kde = KernelDensity(kernel="gaussian", bandwidth=0.5).fit(posterior_samples)
        dens = np.exp(kde.score_samples(posterior_samples))
        threshold = np.quantile(dens, 1 - epsilon)

        # 4. Train baseline SNPE on the same accumulated data
        print("  Training SNPE baseline model...")
        train_snpe(
            snpe_model,
            dataset,
            device,
            epochs=baseline_epochs,
            batch_size=batch_size,
            lr=lr,
            verbose=False,
        )

    # 5. Final sampling with the last trained model
    print("Sampling final posterior with the last trained TSNPSE model...")
    tsnpse_samples = sample_posterior(
        model,
        x_obs,
        device,
        num_samples=5000,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
    )

    # 6. Final sampling with baseline SNPE model
    print("Sampling final posterior with the baseline SNPE model...")
    snpe_samples = sample_snpe(
        snpe_model,
        x_obs,
        device,
        n_samples=5000,
    )

    # 7. Save samples
    np.save(out_path / "tsnpse_samples.npy", tsnpse_samples.astype(np.float32))
    np.save(out_path / "snpe_samples.npy", snpe_samples.astype(np.float32))

    # 8. Compute diagnostics
    mean = tsnpse_samples.mean(axis=0).tolist()
    cov = np.cov(tsnpse_samples, rowvar=False).tolist()
    metrics: Dict = {
        "method": "TSNPSE",
        "mean": mean,
        "covariance": cov,
        "num_simulations": rounds * sims_per_round,
        "num_rounds": rounds,
        "epsilon": epsilon,
    }

    # Baseline diagnostics
    snpe_mean = snpe_samples.mean(axis=0).tolist()
    snpe_cov = np.cov(snpe_samples, rowvar=False).tolist()
    metrics["snpe_mean"] = snpe_mean
    metrics["snpe_covariance"] = snpe_cov

    # For Gaussian linear benchmark, add ground‑truth analytic posterior
    if name == "gaussian_linear":
        # Prior var = 0.1, likelihood var = 0.1
        var_prior = 0.1
        var_lik = 0.1
        mean_post, cov_post = analytic_posterior_gaussian_linear(x_obs, var_prior, var_lik)
        metrics["ground_truth"] = {
            "mean": mean_post.tolist(),
            "covariance": cov_post.numpy().tolist(),
        }

        # Estimate KL divergence using KDE
        kde_true = KernelDensity(kernel="gaussian", bandwidth=0.05).fit(
            np.array([mean_post.cpu().numpy()])  # single point is enough for analytic
        )
        kde_est = KernelDensity(kernel="gaussian", bandwidth=0.5).fit(tsnpse_samples)
        metrics["kl_divergence_to_true"] = kl_divergence_from_kde(tsnpse_samples, kde_true, kde_est)

        # Coverage (per‑dim) and ESS for TSNPSE
        cov_levels = [0.5, 0.8, 0.9, 0.95]
        cov_dict = coverage_per_dim(tsnpse_samples, mean_post.cpu().numpy(), cov_levels)
        metrics["coverage"] = cov_dict
        metrics["ess_tsnpse"] = effective_sample_size(np.ones(tsnpse_samples.shape[0]))
        # Baseline coverage
        snpe_cov_dict = coverage_per_dim(snpe_samples, mean_post.cpu().numpy(), cov_levels)
        metrics["coverage_snpe"] = snpe_cov_dict
        metrics["ess_snpe"] = effective_sample_size(np.ones(snpe_samples.shape[0]))

    with open(out_path / "metrics.json", "w") as fp:
        json.dump(metrics, fp, indent=2)

    print(f"Results saved to {out_path}\n")


def main():
    parser = argparse.ArgumentParser(description="TSNPSE toy experiments with baseline")
    parser.add_argument("--output-dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--rounds", type=int, default=3, help="Number of sequential rounds")
    parser.add_argument("--sims-per-round", type=int, default=2000, help="Simulations per round")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs per round")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epsilon", type=float, default=0.05, help="Truncation epsilon")
    parser.add_argument("--baseline-epochs", type=int, default=10,
                        help="Training epochs for baseline SNPE")
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for bm in BENCHMARKS:
        run_experiment(
            bm,
            output_dir,
            device,
            rounds=args.rounds,
            sims_per_round=args.sims_per_round,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            epsilon=args.epsilon,
            baseline_epochs=args.baseline_epochs,
        )


if __name__ == "__main__":
    main()