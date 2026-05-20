#!/usr/bin/env python3
"""
Inference script that loads a trained Simformer and samples the posterior
p(theta | x_obs) for a new synthetic observation.
"""
import os
import json
import torch
import numpy as np
from tqdm import tqdm
from simformer import Simformer
from utils import sigma_t, target_score
from data import GaussianLinearSimulator

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT = "simformer.pt"
NUM_SAMPLES = 10000
NUM_PARAMS = 5

# -----------------------------
# Load model
# -----------------------------
ckpt = torch.load(CHECKPOINT, map_location='cpu')
model = Simformer(num_vars=ckpt['num_vars'],
                  seq_len=ckpt['seq_len'],
                  embed_dim=64,
                  num_layers=ckpt['num_layers'],
                  nhead=ckpt['nhead'])
model.load_state_dict(ckpt['model_state_dict'])
model.to(DEVICE)
model.eval()

# -----------------------------
# Create a new observation
# -----------------------------
simulator = GaussianLinearSimulator(dim=NUM_PARAMS, noise_std=0.1)
rng = np.random.default_rng(2024)
true_theta = rng.normal(0, np.sqrt(0.1), size=(NUM_PARAMS,))
obs_x = simulator(true_theta)

print(f"True parameters used to generate observation: {true_theta}")
print(f"Observed data x: {obs_x}")

# -----------------------------
# Posterior sampling via reverse diffusion
# -----------------------------
# We'll use a discrete reverse diffusion with 50 steps.
N_STEPS = 50
dt = 1.0 / N_STEPS

# Prepare conditioning mask: data tokens are last NUM_PARAMS indices
COND_INDICES = list(range(NUM_PARAMS, 2*NUM_PARAMS))   # 0‑based
cond_mask = torch.zeros(1, 2*NUM_PARAMS, device=DEVICE)     # (1, SEQ_LEN)
cond_mask[0, COND_INDICES] = 1.0

# Observed values tensor
cond_values = torch.zeros(1, 2*NUM_PARAMS, 1, device=DEVICE)
cond_values[0, COND_INDICES, 0] = torch.tensor(obs_x, dtype=torch.float32, device=DEVICE)

# Start from noise
x = torch.randn(1, 2*NUM_PARAMS, 1, device=DEVICE)

with torch.no_grad():
    for step in tqdm(range(N_STEPS), desc="Reverse diffusion"):
        t = torch.full((1,1), 1.0 - step / N_STEPS, device=DEVICE)  # from 1 to 0
        sigma = sigma_t(t)
        # Predict score
        pred = model(x, cond_mask, t)            # (1, SEQ_LEN)
        # Update
        x = x - sigma.pow(2) * pred.unsqueeze(-1) * dt + sigma * torch.sqrt(dt) * torch.randn_like(x)
        # Re‑fix conditioned tokens
        x[:, COND_INDICES] = cond_values

# After reverse, the first NUM_PARAMS tokens are theta samples
theta_samples = x[0, :NUM_PARAMS, 0].cpu().numpy()
np.save("posterior_samples.npy", theta_samples)

# Summary statistics
mean = theta_samples.mean(axis=0)
std  = theta_samples.std(axis=0)

summary = {
    "num_samples": NUM_SAMPLES,
    "mean": mean.tolist(),
    "std": std.tolist(),
    "true_theta": true_theta.tolist(),
    "observation": obs_x.tolist()
}
with open("posterior_summary.txt", "w") as f:
    json.dump(summary, f, indent=2)

print("Posterior samples and summary saved.")
print("Inference finished successfully.")