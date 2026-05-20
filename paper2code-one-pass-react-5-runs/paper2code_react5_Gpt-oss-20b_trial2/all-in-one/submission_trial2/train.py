#!/usr/bin/env python3
"""
Training script for the Simformer on the Gaussian‑linear simulator.
"""
import os
import torch
import math
from torch.utils.data import DataLoader
from tqdm import tqdm
from data import GaussianLinearDataset, GaussianLinearSimulator
from simformer import Simformer
from utils import sigma_t, target_score

# -----------------------------
# Configuration
# -----------------------------
NUM_PARAMS = 5
SEQ_LEN = NUM_PARAMS * 2          # theta + x
BATCH_SIZE = 128
EPOCHS = 20
LR = 3e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT = "simformer.pt"

# -----------------------------
# Data
# -----------------------------
simulator = GaussianLinearSimulator(dim=NUM_PARAMS, noise_std=0.1)
dataset = GaussianLinearDataset(simulator, n_samples=20000, seed=42)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

# -----------------------------
# Model
# -----------------------------
model = Simformer(num_vars=SEQ_LEN,
                  seq_len=SEQ_LEN,
                  embed_dim=64,
                  num_layers=6,
                  nhead=4).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS*len(dataloader))

# -----------------------------
# Training loop
# -----------------------------
model.train()
for epoch in range(1, EPOCHS+1):
    epoch_loss = 0.0
    for theta, x in tqdm(dataloader, desc=f"Epoch {epoch}/{EPOCHS}"):
        theta = theta.to(DEVICE)   # (B, D)
        x     = x.to(DEVICE)       # (B, D)
        batch = theta.size(0)

        # concatenate theta and x -> (B, SEQ_LEN, 1)
        values = torch.cat([theta, x], dim=1).unsqueeze(-1)

        # conditioning mask: for joint training we use all zeros
        cond = torch.zeros(batch, SEQ_LEN, device=DEVICE)          # (B, SEQ_LEN)

        # sample t uniformly in [0,1]
        t = torch.rand(batch, 1, device=DEVICE)

        # forward diffusion: x_t = x0 + sigma(t) * eps
        sigma = sigma_t(t)
        eps = torch.randn_like(values)
        xt = values + sigma * eps

        # target score
        target = target_score(xt, values, t)

        # forward through model
        pred = model(values, cond, t)          # (B, SEQ_LEN)

        # loss: MSE on unconditioned tokens only (here all tokens)
        loss = F.mse_loss(pred, target.squeeze(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item() * batch

    avg_loss = epoch_loss / len(dataloader.dataset)
    print(f"Epoch {epoch} | Loss: {avg_loss:.6f}")

# -----------------------------
# Save checkpoint
# -----------------------------
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': EPOCHS,
    'seq_len': SEQ_LEN,
    'num_vars': SEQ_LEN,
    'embed_dim': 64,
    'num_layers': 6,
    'nhead': 4
}, CHECKPOINT)

print(f"Training finished. Model saved to {CHECKPOINT}")