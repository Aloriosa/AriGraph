"""
Training loop for the Simformer on the Two‑Moons toy benchmark.
"""

import math
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .data import generate_two_moons
from .model import SimpleTransformerScore, VESDE


def train_and_sample(
    *,
    num_simulations: int = 100_000,
    epochs: int = 10,
    batch_size: int = 1024,
    num_steps: int = 500,
    output_path: str = "samples.csv",
    device: Optional[str] = None,
):
    """
    Train a simple transformer‑based score model on the joint distribution of
    (θ, x) and then generate new samples by reverse diffusion.

    Parameters
    ----------
    num_simulations : int
        How many (θ, x) pairs to generate for training.
    epochs : int
        Number of training epochs.
    batch_size : int
        Training batch size.
    num_steps : int
        Number of reverse diffusion steps.
    output_path : str
        Path to write the generated samples.
    device : str | None
        Device to run on (auto‑detect if None).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1️⃣ Generate the toy dataset
    data_np = generate_two_moons(num_simulations)
    data = torch.from_numpy(data_np).to(device)

    # 2️⃣ Build dataloader
    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # 3️⃣ Instantiate model and optimizer
    model = SimpleTransformerScore(dim=4).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 4️⃣ Diffusion helper
    diffusion = VESDE()

    # 5️⃣ Training loop
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for batch_idx, (x0_batch,) in enumerate(loader):
            x0_batch = x0_batch.to(device)

            # Sample random time t ∈ [0, 1]
            t = torch.rand(x0_batch.size(0), device=device)

            # Forward diffusion
            xt_batch, eps_batch = diffusion.forward_sample(x0_batch, t)

            # Target score: (x0 - xt) / σ(t)^2
            sigma_t = diffusion.sigma(t)
            target_score = (x0_batch - xt_batch) / (sigma_t[:, None] ** 2)

            # Predict score
            pred_score = model(xt_batch)

            loss = F.mse_loss(pred_score, target_score)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * x0_batch.size(0)

        epoch_loss /= len(loader.dataset)
        print(f"Epoch {epoch:02d}/{epochs} – MSE loss: {epoch_loss:.6f}")

        scheduler.step()

    # 6️⃣ Sampling
    model.eval()
    num_samples = 5_000
    with torch.no_grad():
        # Start from Gaussian noise at t = 1
        sigma_T = diffusion.sigma(1.0)
        samples = torch.randn(num_samples, 4, device=device) * sigma_T

        dt = -1.0 / (num_steps - 1)  # negative step size

        for step in range(num_steps - 1):
            t = step / (num_steps - 1)
            samples = diffusion.reverse_step(model, samples, t, dt)

        samples_cpu = samples.cpu().numpy()

    # 7️⃣ Save samples
    np.savetxt(output_path, samples_cpu, delimiter=",")
    print(f"Saved {num_samples} samples to {output_path}")

    # 8️⃣ Quick sanity check
    theta_samples = samples_cpu[:, :2]
    x_samples = samples_cpu[:, 2:]
    print(f"Sample shape: {samples_cpu.shape}")
    print(f"Theta mean / std: {theta_samples.mean():.3f} / {theta_samples.std():.3f}")
    print(f"X     mean / std: {x_samples.mean():.3f} / {x_samples.std():.3f}")