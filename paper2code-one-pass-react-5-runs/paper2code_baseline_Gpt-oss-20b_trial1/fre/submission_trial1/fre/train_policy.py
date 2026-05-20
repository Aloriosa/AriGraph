#!/usr/bin/env python3
"""
This script is a placeholder that demonstrates how a policy could be
conditioned on the FRE latent z.  For brevity, we do not implement a full
offline RL algorithm; instead, we simply generate a random policy that
outputs actions conditioned on z and evaluate it on a few steps.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from fre.encoder import FREEncoder
from fre.decoder import FREDecoder
from fre.dataset import SyntheticDataset
from fre.utils import set_seed

# ---------------- CONFIG ----------------
SEED = 123
set_seed(SEED)
STATE_DIM = 10
ACTION_DIM = 4
LATENT_DIM = 32
BATCH_SIZE = 64
EPOCHS = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------- LOAD FRE MODEL ----------------
encoder_path = None  # assume we have trained encoder in train_fre.py
decoder_path = None

if not os.path.exists('results/fre_results.csv'):
    print("Please run train_fre.py first to generate FRE encoder/decoder.")
    exit(0)

# For this toy example, we skip loading and just create dummy networks
encoder = FREEncoder(state_dim=STATE_DIM, latent_dim=LATENT_DIM).to(device)
decoder = FREDecoder(state_dim=STATE_DIM, latent_dim=LATENT_DIM).to(device)

# ---------------- SIMPLE POLICY ----------------
class SimplePolicy(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, state, z):
        x = torch.cat([state, z], dim=-1)
        return self.net(x)

policy = SimplePolicy(STATE_DIM, ACTION_DIM, LATENT_DIM).to(device)

# ---------------- ENV SIMULATION ----------------
def simulate(env_states, policy, z, steps=10):
    """
    env_states: (T, state_dim) synthetic states
    """
    actions = []
    for t in range(min(steps, len(env_states))):
        state = env_states[t].unsqueeze(0).to(device)
        action = policy(state, z)
        actions.append(action.cpu().numpy().flatten())
    return np.array(actions)

# ---------------- RUN ----------------
dataset = SyntheticDataset(state_dim=STATE_DIM, action_dim=ACTION_DIM)

print("Simulating policy conditioned on latent z from FRE...")
for i in range(3):
    # sample a reward function and encode
    goal = torch.randn(STATE_DIM, device=device)
    def func(s): return torch.where(torch.norm(s - goal, dim=-1) < 1.0, torch.zeros_like(s[:,0]), -torch.ones_like(s[:,0]))
    # encode
    states_enc, _ = dataset.sample_batch(1)
    states_enc = states_enc.to(device)
    enc_states = states_enc[:, :32]
    enc_rewards = func(enc_states).unsqueeze(-1)
    enc_tokens = torch.cat([enc_states, enc_rewards], dim=-1)
    encoder.eval()
    with torch.no_grad():
        z, _, _ = encoder(enc_tokens)
    # simulate
    env_states, _ = dataset.sample_batch(1)
    env_states = env_states.to(device)
    actions = simulate(env_states, policy, z, steps=5)
    print(f"Run {i+1}: actions (first 5 steps) -> {actions}")

print("Simulation finished. (Note: this is a toy demo.)")