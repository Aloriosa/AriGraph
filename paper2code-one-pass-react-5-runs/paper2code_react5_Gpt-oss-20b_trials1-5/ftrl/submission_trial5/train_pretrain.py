#!/usr/bin/env python3
"""
Pre‑train a policy on phase‑2 (return) only using Behavioural Cloning.
Saves the trained policy and the Fisher diagonal (for EWC).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os

# --------------------------- hyper‑parameters --------------------------- #
M = 30          # distance to apple
c = 1.0         # observation bias
max_steps = 100
num_samples = 2000   # number of (obs,action) pairs in the dataset
batch_size = 64
epochs = 20
lr = 1e-2
seed = 42
output_dir = "pretrain"
os.makedirs(output_dir, exist_ok=True)

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# --------------------------- dataset creation --------------------------- #
dataset = []
for _ in range(num_samples):
    # sample a state in phase 2 (x in [1, M])
    x = random.randint(1, M)
    # observation bias in phase 2 is -c (the environment emits -c)
    obs = np.array([-c], dtype=np.float32)
    action = 0  # optimal action is left
    dataset.append((obs, action))

# --------------------------- model & training --------------------------- #
policy = Policy()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(policy.parameters(), lr=lr)

for epoch in range(epochs):
    random.shuffle(dataset)
    epoch_loss = 0.0
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i : i + batch_size]
        obs_batch = torch.tensor([d[0] for d in batch], dtype=torch.float32)
        act_batch = torch.tensor([d[1] for d in batch], dtype=torch.long)
        logits = policy(obs_batch)
        loss = criterion(logits, act_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * len(batch)
    epoch_loss /= len(dataset)
    print(f"Epoch {epoch+1}/{epochs} BC loss: {epoch_loss:.4f}")

# --------------------------- Fisher diagonal (EWC) --------------------------- #
print("Computing Fisher diagonal for EWC...")
fisher = []
for p in policy.parameters():
    fisher.append(torch.zeros_like(p.data))

# compute gradient of log‑likelihood for each sample
for obs, act in dataset:
    obs_t = torch.tensor([obs], dtype=torch.float32, requires_grad=True)
    logits = policy(obs_t)
    log_probs = nn.functional.log_softmax(logits, dim=-1)
    log_prob = log_probs[0, act]
    grads = torch.autograd.grad(log_prob, policy.parameters(), retain_graph=False, allow_unused=True)
    for idx, g in enumerate(grads):
        if g is not None:
            fisher[idx].add_(g.detach() ** 2)

# average over samples
fisher = [f / len(dataset) for f in fisher]
# save
torch.save({
    "policy_state_dict": policy.state_dict(),
    "fisher": fisher,
    "hparams": {"M": M, "c": c, "max_steps": max_steps, "seed": seed}
}, os.path.join(output_dir, "pretrain.pt"))

print("Pre‑training finished. Model saved to", os.path.join(output_dir, "pretrain.pt"))