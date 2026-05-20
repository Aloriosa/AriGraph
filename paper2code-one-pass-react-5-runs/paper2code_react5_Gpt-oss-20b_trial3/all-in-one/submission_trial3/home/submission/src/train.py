#!/usr/bin/env python
"""
Training script for Simformer on a given task.
"""

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from .model import Simformer
from .diffusion import VESDE
from .utils import random_attention_mask, identity_attention_mask, full_attention_mask
from ..tasks.gaussian_linear import GaussianLinearSim
from ..tasks.two_moons import TwoMoonsSim

# ---------------------------------------------------------------------------

def get_task(task_name):
    if task_name == "gaussian_linear":
        return GaussianLinearSim()
    if task_name == "two_moons":
        return TwoMoonsSim()
    raise ValueError(f"Unknown task {task_name}")


def random_condition_mask(batch_size, prob_joint=0.25, prob_posterior=0.25,
                          prob_likelihood=0.25, prob_random=0.25, data_dim=2):
    """
    Sample a condition mask according to the paper:
        - joint:   all zeros
        - posterior: data conditioned, params not
        - likelihood: params conditioned, data not
        - random: two random binary vectors
    Returns:
        mask: [batch, 2]  (params, data)
    """
    masks = []
    for _ in range(batch_size):
        r = random.random()
        if r < prob_joint:
            masks.append([0, 0])
        elif r < prob_joint + prob_posterior:
            masks.append([0, 1])
        elif r < prob_joint + prob_posterior + prob_likelihood:
            masks.append([1, 0])
        else:
            # random two binary masks
            masks.append([random.randint(0, 1), random.randint(0, 1)])
    return torch.tensor(masks, dtype=torch.float32)


def train(
    task_name: str,
    epochs: int = 50,
    batch_size: int = 512,
    lr: float = 1e-3,
    embed_dim: int = 64,
    n_layers: int = 6,
    n_heads: int = 4,
    attention_mask_type: str = "full",
    use_fourier: bool = False,
    fourier_dim: int = 32,
    device: str = "cpu",
):
    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    task = get_task(task_name)
    param_dim, data_dim = task.param_dim, task.data_dim

    model = Simformer(
        param_dim,
        data_dim,
        embed_dim=embed_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        use_fourier=use_fourier,
        fourier_dim=fourier_dim,
        attention_mask_type=attention_mask_type,
    ).to(device)

    diffusion = VESDE()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss(reduction="none")  # keep per‑sample loss

    model.train()
    for epoch in range(epochs):
        pbar = tqdm(range(0, task.num_samples, batch_size), desc=f"Epoch {epoch+1}")
        for i in pbar:
            # Sample batch from simulator
            theta, x = task.sample_batch(batch_size)  # [batch, dim]
            theta = theta.to(device)
            x = x.to(device)

            # Random condition mask
            cond_mask = random_condition_mask(
                batch_size, data_dim=data_dim
            ).to(device)  # [batch, 2]

            # Diffusion time t ∈ (0,1]
            t = torch.rand(batch_size, device=device)

            # Compute noisy sample
            sigma_t = diffusion.sigma(t)
            eps = torch.randn_like(theta)
            # Apply noise only to unconditioned variables
            mask_theta = cond_mask[:, 0].unsqueeze(-1)  # [batch,1]
            mask_x = cond_mask[:, 1].unsqueeze(-1)
            theta_noisy = theta + sigma_t.unsqueeze(-1) * eps * (1 - mask_theta)
            x_noisy = x + sigma_t.unsqueeze(-1) * eps * (1 - mask_x)

            # Predict scores
            pred_scores = model(theta_noisy, x_noisy, cond_mask, t)  # [batch,2,embed_dim]

            # Target scores
            target_theta = diffusion.score(theta_noisy[:, 0:1], theta[:, 0:1], t)
            target_x = diffusion.score(x_noisy[:, 0:1], x[:, 0:1], t)
            # Expand to match embed_dim
            target_theta = target_theta.repeat(1, embed_dim)
            target_x = target_x.repeat(1, embed_dim)

            target = torch.zeros_like(pred_scores)
            target[:, 0] = target_theta
            target[:, 1] = target_x

            loss = mse_loss(pred_scores, target)
            # Mask out conditioned variables
            loss = loss * (1 - cond_mask.unsqueeze(-1))
            loss = loss.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    # Save model
    os.makedirs("models", exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "task": task_name,
            "param_dim": param_dim,
            "data_dim": data_dim,
            "embed_dim": embed_dim,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "attention_mask_type": attention_mask_type,
            "use_fourier": use_fourier,
            "fourier_dim": fourier_dim,
        },
        f"models/{task_name}_simformer.pth",
    )
    print(f"Model saved to models/{task_name}_simformer.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    train(**vars(args))