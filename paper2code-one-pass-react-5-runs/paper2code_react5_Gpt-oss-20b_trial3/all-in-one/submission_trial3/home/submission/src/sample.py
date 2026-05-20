#!/usr/bin/env python
"""
Sampling script for Simformer.

Generates posterior samples for a held‑out observation and saves them as CSV.
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .model import Simformer
from .diffusion import VESDE
from ..tasks.gaussian_linear import GaussianLinearSim
from ..tasks.two_moons import TwoMoonsSim


def load_model(path):
    ckpt = torch.load(path, map_location="cpu")
    model = Simformer(
        ckpt["param_dim"],
        ckpt["data_dim"],
        embed_dim=ckpt["embed_dim"],
        n_layers=ckpt["n_layers"],
        n_heads=ckpt["n_heads"],
        attention_mask_type=ckpt["attention_mask_type"],
        use_fourier=ckpt["use_fourier"],
        fourier_dim=ckpt["fourier_dim"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def sample_posterior(
    model,
    diffusion: VESDE,
    theta_true=None,
    x_true=None,
    n_samples=2000,
    device="cpu",
    guidance=None,
):
    """
    Sample from the posterior p(theta | x_obs) by conditioning on observed data.
    """
    # Setup conditioning mask: data observed (1), params unobserved (0)
    cond_mask = torch.tensor([[0, 1]], dtype=torch.float32, device=device).repeat(
        n_samples, 1
    )

    # Observation (data) – will be copied into each sample
    if x_true is None:
        raise ValueError("x_true must be provided")
    x_obs = torch.tensor(x_true, dtype=torch.float32, device=device).unsqueeze(0)
    x_obs = x_obs.repeat(n_samples, 1)

    # Initialize at terminal distribution
    # x_T ~ N(0, sigma_T^2) for each variable
    sigma_T = diffusion.sigma(1.0)
    # For params: we sample from standard normal (no scaling)
    theta_T = torch.randn(n_samples, model.tokenizer.param_dim, device=device) * sigma_T
    # For data: we sample from standard normal as well
    x_T = torch.randn(n_samples, model.tokenizer.data_dim, device=device) * sigma_T

    # Reverse diffusion
    steps = 50
    dt = 1.0 / steps
    t_vals = torch.linspace(1.0, 0.0, steps + 1, device=device)

    theta_curr = theta_T
    x_curr = x_T

    for i in tqdm(range(steps), desc="Sampling"):
        t = t_vals[i]
        t_next = t_vals[i + 1]
        # Compute scores
        pred_scores = model(theta_curr, x_curr, cond_mask, t * torch.ones(n_samples, device=device))

        # Extract scores for params (first token)
        s_theta = pred_scores[:, 0, :]  # [batch, embed_dim]
        # We need a scalar score per variable; we average across embedding dim
        s_theta = s_theta.mean(dim=-1, keepdim=True)  # [batch,1]

        # Guidance: optional adjustment of score
        if guidance is not None:
            s_theta = guidance(s_theta, theta_curr, x_curr, t)

        # Euler–Maruyama reverse step
        sigma_t = diffusion.sigma(t)
        eps = torch.randn_like(theta_curr)
        theta_curr = theta_curr + (-sigma_t**2 * s_theta) * dt + sigma_t * torch.sqrt(dt) * eps

        # Keep data conditioned
        x_curr = x_obs  # fixed

    # After loop, theta_curr is our posterior sample
    return theta_curr.cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=2000)
    args = parser.parse_args()

    # Load task to get a held‑out observation
    if args.task == "gaussian_linear":
        from ..tasks.gaussian_linear import GaussianLinearSim

        task = GaussianLinearSim()
        # Sample one observation
        theta_true, x_true = task.sample_batch(1)
    elif args.task == "two_moons":
        from ..tasks.two_moons import TwoMoonsSim

        task = TwoMoonsSim()
        theta_true, x_true = task.sample_batch(1)
    else:
        raise ValueError(f"Unknown task {args.task}")

    # Load model
    model_path = f"models/{args.task}_simformer.pth"
    model = load_model(model_path).to("cpu")
    diffusion = VESDE()

    # Sample posterior
    theta_samples = sample_posterior(
        model,
        diffusion,
        theta_true=theta_true.numpy(),
        x_true=x_true.numpy(),
        n_samples=args.n_samples,
    )

    # Save to CSV
    os.makedirs("results", exist_ok=True)
    out_path = f"results/{args.task}_posterior.csv"
    np.savetxt(out_path, theta_samples, delimiter=",")
    print(f"Posterior samples saved to {out_path}")

    # For predictive, we generate new data from sampled params
    # We use the simulator to generate data
    data_samples = []
    for theta in theta_samples:
        _, x = task.sample_batch(1, theta=theta)
        data_samples.append(x.squeeze(0))
    data_samples = np.vstack(data_samples)

    out_path = f"results/{args.task}_predictive.csv"
    np.savetxt(out_path, data_samples, delimiter=",")
    print(f"Predictive samples saved to {out_path}")


if __name__ == "__main__":
    main()