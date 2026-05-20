#!/usr/bin/env python3
"""
Training loop for FRE encoder and Q‑policy on CartPole.
"""
import os
import random
import pickle
import math
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm

from models import RewardEncoder, QNetwork
from utils import (
    OfflineDataset,
    sample_random_reward,
    GoalReachReward,
    LinearReward,
    MLPReward,
)

# ---------- Hyperparameters ----------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 256
EPOCHS = 5
K_ENCODE = 8          # number of (state, reward) pairs for encoding
K_DECOD = 4           # number of states for decoding (not used in this toy)
LEARNING_RATE = 1e-3
DISCOUNT = 0.99
KL_WEIGHT = 1e-2

DATA_FILE = "offline_data.pkl"
ENCODER_FILE = "encoder.pt"
QFILE = "q.pt"

def load_dataset():
    with open(DATA_FILE, "rb") as f:
        transitions = pickle.load(f)
    return OfflineDataset(transitions)

def train():
    dataset = load_dataset()
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Model init
    state_dim = 4  # CartPole
    encoder = RewardEncoder(state_dim=state_dim).to(DEVICE)
    qnet = QNetwork(state_dim=state_dim, latent_dim=32).to(DEVICE)
    target_qnet = QNetwork(state_dim=state_dim, latent_dim=32).to(DEVICE)
    target_qnet.load_state_dict(qnet.state_dict())

    enc_opt = optim.Adam(encoder.parameters(), lr=LEARNING_RATE)
    q_opt = optim.Adam(qnet.parameters(), lr=LEARNING_RATE)

    mse = nn.MSELoss()

    for epoch in range(EPOCHS):
        pbar = tqdm.tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch in pbar:
            states, actions, next_states, _ = [x.to(DEVICE) for x in batch]

            # ----- Sample reward function and encode -----
            reward_fns = [sample_random_reward(state_dim) for _ in range(states.size(0))]
            # For each example in the batch, sample K_ENCODE states from the dataset
            # (here we reuse the current batch states for simplicity)
            # Compute rewards for these states
            enc_states = states[:, :K_ENCODE, :]  # [B, K, state_dim]
            # Compute rewards
            rewards = []
            for i, rf in enumerate(reward_fns):
                r = torch.tensor([rf(s.cpu().numpy()) for s in enc_states[i]], dtype=torch.float32)
                rewards.append(r)
            rewards = torch.stack(rewards, dim=0).unsqueeze(-1)  # [B, K, 1]

            # Encode
            z, mean, logvar = encoder(enc_states, rewards)

            # ----- Decoder loss (MSE between predicted and true rewards) -----
            # For simplicity, we skip an explicit decoder; we only train encoder to
            # predict rewards for the same K_ENCODE samples (auto‑encoding).
            pred_r = torch.matmul(enc_states, mean.unsqueeze(-1)).squeeze(-1)  # dummy
            # Use the true rewards as targets
            decoder_loss = mse(pred_r, rewards.squeeze(-1))

            # KL penalty
            kl_loss = KL_WEIGHT * torch.mean(-0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=-1))

            enc_loss = decoder_loss + kl_loss
            enc_opt.zero_grad()
            enc_loss.backward()
            enc_opt.step()

            # ----- Q‑network update (IQL‑style) -----
            with torch.no_grad():
                next_q = target_qnet(next_states, z).max(dim=1)[0]
                target_q = reward_fns[0].__call__(states.squeeze(0).cpu().numpy())  # dummy reward
                target_q = torch.tensor(target_q, dtype=torch.float32).to(DEVICE)
                target = target_q + DISCOUNT * next_q

            pred_q = qnet(states, z).gather(1, actions.unsqueeze(-1)).squeeze(-1)
            q_loss = mse(pred_q, target)

            q_opt.zero_grad()
            q_loss.backward()
            q_opt.step()

            # Soft target update
            for p, t in zip(qnet.parameters(), target_qnet.parameters()):
                t.data.copy_(0.995 * t.data + 0.005 * p.data)

            pbar.set_postfix(
                enc=float(enc_loss.item()), q=float(q_loss.item()), kl=float(kl_loss.item())
            )

    # Save models
    torch.save(encoder.state_dict(), ENCODER_FILE)
    torch.save(qnet.state_dict(), QFILE)
    print(f"Training finished – models saved: {ENCODER_FILE}, {QFILE}")

if __name__ == "__main__":
    train()