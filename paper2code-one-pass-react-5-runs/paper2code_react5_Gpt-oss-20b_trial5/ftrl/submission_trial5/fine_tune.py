#!/usr/bin/env python3
"""
Fine‑tune the policy on the full AppleRetrieval task using REINFORCE.
Optionally apply knowledge‑retention: BC (Behavioural Cloning) or EWC.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import argparse
import os
from apple_retrieval import AppleRetrieval
from policy import Policy

# --------------------------- hyper‑parameters --------------------------- #
M = 30
c = 1.0
max_steps = 100
seed = 42
pretrain_path = "pretrain/pretrain.pt"
output_dir = "finetune"
os.makedirs(output_dir, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--method", type=str, default="BC", choices=["none", "BC", "EWC"],
                    help="Knowledge retention method to use (none, BC, EWC)")
parser.add_argument("--episodes", type=int, default=500, help="Number of training episodes")
parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
parser.add_argument("--bc_weight", type=float, default=1.0, help="Weight of BC loss")
parser.add_argument("--ewc_lambda", type=float, default=2e6, help="EWC regularization strength")
args = parser.parse_args()

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# load pre‑trained policy
pretrain = torch.load(pretrain_path, map_location="cpu")
pretrain_policy = Policy()
pretrain_policy.load_state_dict(pretrain["policy_state_dict"])
pretrain_policy.eval()
fisher = pretrain["fisher"]

# set up finetune policy
finetune_policy = Policy()
optimizer = optim.Adam(finetune_policy.parameters(), lr=args.lr)

# create BC buffer from pre‑training data (phase‑2 states)
bc_dataset = []
for _ in range(2000):
    # sample a state in phase 2 (x in [1, M])
    x = random.randint(1, M)
    obs = np.array([-c], dtype=np.float32)   # bias in phase 2
    action = 0
    bc_dataset.append((obs, action))

# --------------------------- training loop --------------------------- #
def run_episode(env, policy, deterministic=False):
    obs, _ = env.reset()
    traj = []
    done = False
    while not done:
        action = policy.get_action(obs, deterministic=deterministic)
        next_obs, reward, done, _ = env.step(action)
        traj.append((obs, action, reward))
        obs = next_obs
    return traj

for ep in range(args.episodes):
    env = AppleRetrieval(M=M, c=c, max_steps=max_steps, seed=seed + ep)
    traj = run_episode(env, finetune_policy)
    # compute discounted returns
    returns = []
    G = 0.0
    for _, _, r in reversed(traj):
        G = r + args.gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32)

    # policy gradient loss
    policy_loss = 0.0
    for (obs, action, _), G in zip(traj, returns):
        obs_t = torch.tensor([obs], dtype=torch.float32)
        logits = finetune_policy(obs_t)
        log_prob = nn.functional.log_softmax(logits, dim=-1)[0, action]
        policy_loss -= log_prob * G
    policy_loss = policy_loss / len(traj)

    # knowledge retention losses
    if args.method == "BC":
        # sample a minibatch from BC buffer
        batch = random.sample(bc_dataset, 32)
        obs_batch = torch.tensor([b[0] for b in batch], dtype=torch.float32)
        act_batch = torch.tensor([b[1] for b in batch], dtype=torch.long)
        logits = finetune_policy(obs_batch)
        # KL(pretrain || current)
        with torch.no_grad():
            pre_logits = pretrain_policy(obs_batch)
            pre_probs = torch.softmax(pre_logits, dim=-1)
        current_probs = torch.softmax(logits, dim=-1)
        kl = (pre_probs * (pre_probs.log() - torch.log(current_probs + 1e-10))).sum(-1).mean()
        total_loss = policy_loss + args.bc_weight * kl
    elif args.method == "EWC":
        # EWC penalty
        ewc_loss = 0.0
        for idx, (param, param_pre, fisher_param) in enumerate(zip(finetune_policy.parameters(),
                                                                  pretrain_policy.parameters(),
                                                                  fisher)):
            ewc_loss += (fisher_param * (param - param_pre).pow(2)).sum()
        total_loss = policy_loss + args.ewc_lambda * ewc_loss
    else:
        total_loss = policy_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if (ep + 1) % 50 == 0:
        print(f"Episode {ep+1}/{args.episodes} | Policy loss: {policy_loss.item():.4f} | "
              f"Total loss: {total_loss.item():.4f}")

# --------------------------- saving --------------------------- #
torch.save({
    "policy_state_dict": finetune_policy.state_dict(),
    "pretrain_policy_state_dict": pretrain["policy_state_dict"],
    "method": args.method,
    "hparams": {"M": M, "c": c, "max_steps": max_steps, "seed": seed}
}, os.path.join(output_dir, "finetune.pt"))

print("Fine‑tuning finished. Model saved to", os.path.join(output_dir, "finetune.pt"))