#!/usr/bin/env python3
"""
Sequential Neural Posterior Score Estimation (TSNPSE) – Toy implementation.
"""

import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from scipy.stats import chi2
from sklearn.metrics import pairwise_distances
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data_utils import simulator
from diffusion import sigma_t
from score_network import ScoreNetwork
from utils import set_seed

# --------------------------------------------------------------------------- #
# Dataset and utilities
# --------------------------------------------------------------------------- #
class TrainDataset(Dataset):
    def __init__(self, data):
        self.data = data  # list of tuples (theta_t, x, t, target)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        theta_t, x, t, target = self.data[idx]
        return theta_t, x, t, target


def sample_from_prior(num_samples, prior_sampler):
    """Sample from the prior distribution."""
    return prior_sampler(num_samples)


def sample_truncated_prior(
    num_samples,
    prior_sampler,
    mean,
    cov,
    epsilon,
    device,
):
    """
    Sample from the truncated prior defined by the (1‑ε) HPR
    of a Gaussian with given mean and covariance.
    We approximate the HPR by an ellipsoid derived from the
    Mahalanobis distance.
    """
    d = mean.shape[0]
    threshold = chi2.ppf(1.0 - epsilon, d)
    inv_cov = torch.linalg.pinv(cov)

    samples = []
    batch_size = max(num_samples * 2, 1024)
    while len(samples) < num_samples:
        batch = prior_sampler(batch_size).to(device)
        diff = batch - mean
        mdist = torch.sum(diff @ inv_cov * diff, dim=1)
        mask = mdist <= threshold
        accepted = batch[mask]
        samples.extend(accepted.tolist())
    samples = torch.stack(samples[:num_samples], dim=0)
    return samples


def compute_mean_cov(samples):
    """Compute mean and covariance of a tensor of samples."""
    mean = samples.mean(dim=0)
    cov = torch.from_numpy(
        np.cov(samples.cpu().numpy(), rowvar=False)
    ).float()
    return mean, cov


def sample_posterior(
    num_samples,
    x_obs,
    network,
    sigma_min,
    sigma_max,
    device,
    num_steps=200,
):
    """Sample from the posterior using the probability‑flow ODE."""
    network.eval()
    with torch.no_grad():
        # Start from reference distribution N(0, sigma_max^2)
        theta = torch.randn(num_samples, x_obs.shape[0], device=device) * sigma_max
        ts = torch.linspace(1.0, 0.0, steps=num_steps, device=device)
        dt = -1.0 / (num_steps - 1)

        for t in ts:
            sigma = sigma_t(t.item(), sigma_min, sigma_max)
            # Expand x_obs to batch
            x_batch = x_obs.expand(num_samples, -1)
            t_batch = torch.full((num_samples, 1), t, device=device)
            score = network(theta, x_batch, t_batch)
            v = -0.5 * (sigma**2) * score
            theta = theta + v * dt
        return theta.cpu()


def plot_samples(samples, filename="posterior_samples.png"):
    """Scatter plot of the posterior samples."""
    if samples.shape[1] != 2:
        return
    plt.figure(figsize=(5, 5))
    plt.scatter(samples[:, 0], samples[:, 1], s=10, alpha=0.5)
    plt.title("Posterior Samples")
    plt.xlabel("$\\theta_1$")
    plt.ylabel("$\\theta_2$")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


# --------------------------------------------------------------------------- #
# Main routine
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="TSNPSE toy demo")
    parser.add_argument("--dataset", type=str, default="two_moons")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--samples-per-round", type=int, default=2000)
    parser.add_argument("--sigma-min", type=float, default=0.01)
    parser.add_argument("--sigma-max", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Prior: standard normal N(0, I)
    prior_sampler = lambda n: torch.randn(n, 2, device=device)

    # Observation: generate a single point from the true simulator
    true_theta = torch.tensor([0.5, -0.3], dtype=torch.float32, device=device)
    x_obs = simulator(true_theta, n_samples=1).squeeze(0)

    # Score network
    network = ScoreNetwork(theta_dim=2, x_dim=2).to(device)

    # Accumulate training data across rounds
    training_data = []

    for r in range(1, args.rounds + 1):
        print(f"\n=== Round {r} ===")

        # Sample parameters
        if r == 1:
            theta0_samples = sample_from_prior(args.samples_per_round, prior_sampler)
        else:
            mean, cov = compute_mean_cov(posterior_samples)
            theta0_samples = sample_truncated_prior(
                args.samples_per_round,
                prior_sampler,
                mean,
                cov,
                args.epsilon,
                device,
            )

        # Simulate data and generate training examples
        for theta0 in theta0_samples:
            # Simulate observation x
            x = simulator(theta0, n_samples=1).squeeze(0)

            # Random time t
            t = random.random()
            sigma = sigma_t(t, args.sigma_min, args.sigma_max)

            # Add noise
            noise = torch.randn_like(theta0, device=device) * sigma
            theta_t = theta0 + noise
            target_score = (theta0 - theta_t) / (sigma**2)

            training_data.append((theta_t, x, torch.tensor([t], dtype=torch.float32), target_score))

        # Train score network on accumulated data
        train_loader = DataLoader(
            TrainDataset(training_data), batch_size=args.batch_size, shuffle=True
        )
        optimizer = optim.Adam(network.parameters(), lr=1e-4)
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(args.max_epochs):
            network.train()
            epoch_loss = 0.0
            for theta_t, x, t, target in train_loader:
                theta_t = theta_t.to(device)
                x = x.to(device)
                t = t.to(device)
                target = target.to(device)

                optimizer.zero_grad()
                pred = network(theta_t, x, t)
                loss = nn.functional.mse_loss(pred, target)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * theta_t.size(0)

            epoch_loss /= len(train_loader.dataset)
            print(f"  Epoch {epoch+1:02d} – loss: {epoch_loss:.6f}")

            # Validation (use a small hold‑out set)
            if len(train_loader.dataset) > 0:
                val_loss = epoch_loss
                if val_loss < best_val_loss - 1e-6:
                    best_val_loss = val_loss
                    torch.save(network.state_dict(), "best_network.pt")
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= args.patience:
                        print("  Early stopping")
                        break

        # Load best model
        network.load_state_dict(torch.load("best_network.pt"))

        # Sample posterior to compute HPR for next round
        posterior_samples = sample_posterior(
            args.samples_per_round,
            x_obs,
            network,
            args.sigma_min,
            args.sigma_max,
            device,
        )
        mean, cov = compute_mean_cov(posterior_samples)
        print(f"  Posterior mean: {mean.numpy()}")
        print(f"  Posterior covariance:\n{cov.numpy()}")

    # Final posterior
    final_samples = sample_posterior(
        5000,
        x_obs,
        network,
        args.sigma_min,
        args.sigma_max,
        device,
        num_steps=300,
    )
    np.save("posterior_samples.npy", final_samples.numpy())
    plot_samples(final_samples.numpy())

    print("\n=== Final Posterior ===")
    mean, cov = compute_mean_cov(final_samples)
    print(f"Mean: {mean.numpy()}")
    print(f"Covariance:\n{cov.numpy()}")


if __name__ == "__main__":
    main()