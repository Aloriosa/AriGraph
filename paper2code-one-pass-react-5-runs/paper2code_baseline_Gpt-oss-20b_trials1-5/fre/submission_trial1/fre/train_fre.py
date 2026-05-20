#!/usr/bin/env python3
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

from fre.utils import set_seed
from fre.dataset import SyntheticDataset
from fre.encoder import FREEncoder
from fre.decoder import FREDecoder

# ---------------------- CONFIG ----------------------
SEED = 42
set_seed(SEED)

STATE_DIM = 10
ACTION_DIM = 4
LATENT_DIM = 32
BATCH_SIZE = 64
ENC_K = 32      # number of encoder state–reward pairs
DEC_K = 8       # number of decoder state–reward pairs
EPOCHS = 20
LR = 1e-3
BETA = 0.01     # KL weight

# reward prior mixture weights
PROB_GOAL = 0.33
PROB_LINEAR = 0.33
PROB_MLP = 0.34

# ---------------------- DATA ----------------------
dataset = SyntheticDataset(state_dim=STATE_DIM, action_dim=ACTION_DIM)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------- MODEL ----------------------
encoder = FREEncoder(state_dim=STATE_DIM, latent_dim=LATENT_DIM).to(device)
decoder = FREDecoder(state_dim=STATE_DIM, latent_dim=LATENT_DIM).to(device)

optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=LR)

# ---------------------- REWARD FUNCS ----------------------
def random_goal_reward(state, goal):
    """-1 for all states except 0.0 reward if within 1.0 of goal."""
    dist = torch.norm(state - goal, dim=-1)
    return torch.where(dist < 1.0, torch.zeros_like(dist), -torch.ones_like(dist))

def random_linear_reward(state):
    w = torch.randn(STATE_DIM, device=device)  # fixed per call
    return torch.matmul(state, w)

def random_mlp_reward(state):
    # simple 2‑layer MLP
    w1 = torch.randn(STATE_DIM, 32, device=device)
    b1 = torch.randn(32, device=device)
    w2 = torch.randn(32, 1, device=device)
    b2 = torch.randn(1, device=device)
    h = torch.tanh(state @ w1 + b1)
    return (h @ w2 + b2).squeeze(-1)

def sample_reward_func():
    r = np.random.rand()
    if r < PROB_GOAL:
        goal = torch.randn(STATE_DIM, device=device)
        return lambda s: random_goal_reward(s, goal), 'goal', goal
    elif r < PROB_GOAL + PROB_LINEAR:
        return lambda s: random_linear_reward(s), 'linear', None
    else:
        return lambda s: random_mlp_reward(s), 'mlp', None

# ---------------------- TRAIN LOOP ----------------------
def train_one_epoch():
    encoder.train()
    decoder.train()
    total_loss = 0.0
    for _ in tqdm(range(len(dataset.states) // BATCH_SIZE), desc='Epoch'):
        # sample batch of states
        states, _ = dataset.sample_batch(BATCH_SIZE)
        states = states.to(device)

        # sample a reward function
        func, rtype, goal = sample_reward_func()

        # compute rewards for encoder set
        enc_states = states[:, :ENC_K]          # (B, K, state_dim)
        enc_rewards = func(enc_states).unsqueeze(-1)  # (B, K, 1)
        enc_tokens = torch.cat([enc_states, enc_rewards], dim=-1)  # (B, K, token_dim)

        # compute rewards for decoder set
        dec_states = states[:, ENC_K:ENC_K+DEC_K]
        dec_rewards = func(dec_states).unsqueeze(-1)

        # forward pass
        z, mean, logvar = encoder(enc_tokens)          # (B, latent_dim)
        pred_rewards = decoder(dec_states, z)          # (B, DEC_K)

        # reconstruction loss
        recon_loss = nn.functional.mse_loss(pred_rewards, dec_rewards.squeeze(-1))

        # KL loss
        kl = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=-1).mean()

        loss = recon_loss + BETA * kl

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / (len(dataset.states) // BATCH_SIZE)

print("Starting FRE training...")
for epoch in range(1, EPOCHS+1):
    loss = train_one_epoch()
    print(f"Epoch {epoch:02d} loss: {loss:.4f}")

# ---------------------- EVALUATION ----------------------
os.makedirs('results', exist_ok=True)
out_path = 'results/fre_results.csv'
with open(out_path, 'w') as f:
    f.write('reward_type,goal_vector,latent_z,decoded_reward_mean\n')
    # evaluate on 5 random reward functions
    for i in range(5):
        func, rtype, goal = sample_reward_func()
        # sample 32 states for encoding
        states_enc, _ = dataset.sample_batch(1)
        states_enc = states_enc.to(device)
        enc_states = states_enc[:, :ENC_K]
        enc_rewards = func(enc_states).unsqueeze(-1)
        enc_tokens = torch.cat([enc_states, enc_rewards], dim=-1)
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            z, _, _ = encoder(enc_tokens)
            # sample 8 states for decoding
            states_dec, _ = dataset.sample_batch(1)
            states_dec = states_dec.to(device)
            dec_states = states_dec[:, ENC_K:ENC_K+DEC_K]
            preds = decoder(dec_states, z)
            pred_mean = preds.mean().item()

        z_np = z.cpu().numpy().flatten()
        goal_str = 'NA' if goal is None else np.round(goal.cpu().numpy(), 3).tolist()
        f.write(f'{rtype},{goal_str},{z_np.tolist()},{pred_mean:.4f}\n')

print(f"Results written to {out_path}")