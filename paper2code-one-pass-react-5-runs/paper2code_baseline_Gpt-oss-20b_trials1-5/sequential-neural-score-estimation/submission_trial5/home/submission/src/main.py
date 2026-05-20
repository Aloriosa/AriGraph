#!/usr/bin/env python3
"""
Toy NPSE implementation for a Gaussian‑linear simulator.

Posterior: p(θ | x) with θ ~ N(0, 0.1I),  x | θ ~ N(θ, 0.1I)
"""

import os
import torch
import numpy as np
import pandas as pd
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from utils import DiffusionParams, forward_diffusion, target_score, sample_posterior
from model import MLPScore

# ----------------------------------------------------------------------
#  Configuration
# ----------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 512
N_SAMPLES = 10000          # training examples
N_EPOCHS = 5
LEARNING_RATE = 1e-4
N_PREDICT_SAMPLES = 5000   # posterior samples to draw
N_STEPS = 2000             # ODE steps
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# ----------------------------------------------------------------------
#  Synthetic simulator
# ----------------------------------------------------------------------
def simulator(theta: torch.Tensor) -> torch.Tensor:
    """
    x | θ ~ N(θ, 0.1I)
    """
    noise = torch.randn_like(theta) * math.sqrt(0.1)
    return theta + noise

# ----------------------------------------------------------------------
#  Data generation
# ----------------------------------------------------------------------
def generate_batch(batch_size: int):
    theta0 = torch.randn(batch_size, 2, device=DEVICE) * math.sqrt(0.1)
    x = simulator(theta0)
    # Sample t uniformly from [0,1]
    t = torch.rand(batch_size, 1, device=DEVICE)
    # Forward diffusion
    theta_t = forward_diffusion(theta0, t, diffusion)
    # Target score
    s_target = target_score(theta_t, theta0, diffusion)
    return theta_t, x, t, s_target

# ----------------------------------------------------------------------
#  Training
# ----------------------------------------------------------------------
diffusion = DiffusionParams(sigma_min=0.01, sigma_max=1.0, T=1.0)
model = MLPScore(d_theta=2, d_x=2).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

print("Starting training...")

for epoch in range(N_EPOCHS):
    epoch_loss = 0.0
    for _ in range(N_SAMPLES // BATCH_SIZE):
        theta_t, x, t, s_target = generate_batch(BATCH_SIZE)
        optimizer.zero_grad()
        s_pred = model(theta_t, x, t)
        loss = criterion(s_pred, s_target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}/{N_EPOCHS}  loss={epoch_loss / (N_SAMPLES // BATCH_SIZE):.6f}")

print("Training finished.")

# ----------------------------------------------------------------------
#  Posterior sampling
# ----------------------------------------------------------------------
x_obs = torch.zeros(2, device=DEVICE)   # observation x = [0,0]
posterior_samples = sample_posterior(
    score_net=model,
    diffusion=diffusion,
    x_obs=x_obs,
    n_samples=N_PREDICT_SAMPLES,
    n_steps=N_STEPS,
    device=DEVICE
)

# ----------------------------------------------------------------------
#  Save results
# ----------------------------------------------------------------------
samples_np = posterior_samples.cpu().numpy()
df = pd.DataFrame(samples_np, columns=['θ₁', 'θ₂'])
df.to_csv('posterior_samples.csv', index=False)
print(f"Saved {N_PREDICT_SAMPLES} posterior samples to posterior_samples.csv")