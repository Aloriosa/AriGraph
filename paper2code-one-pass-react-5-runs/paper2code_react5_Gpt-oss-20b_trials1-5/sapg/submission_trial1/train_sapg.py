#!/usr/bin/env python3
"""
Main training script for SAPG on Pendulum-v1.
"""

import os
import random
import numpy as np
import torch
import gymnasium as gym
from sapg import SAPGAgent
from env_utils import collect_trajectories

# Set deterministic seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Environment and hyper-parameters
env_name = "Pendulum-v1"
num_envs = 32          # Total parallel environments (increase for large‑scale tests)
M = 4                  # Number of policies (leader + 3 followers)
horizon = 64           # Steps per rollout per env
num_epochs = 10
eval_episodes = 5

# Create directories
os.makedirs("logs", exist_ok=True)

# Create environments
envs = [gym.make(env_name, render_mode=None) for _ in range(num_envs)]
for idx, env in enumerate(envs):
    env.action_space.seed(SEED + idx)
    env.observation_space.seed(SEED + idx)

# Shared device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate SAPG agent
agent = SAPGAgent(
    envs[0].observation_space.shape[0],
    envs[0].action_space.shape[0],
    M=M,
    horizon=horizon,
    gamma=0.99,
    tau=0.95,
    lr=5e-4,
    eps_clip=0.2,
    lambda_off=1.0,
    entropy_coef=[0.0, 0.003, 0.005, 0.0],  # Leader gets 0
    device=device,
)

# Training loop
for epoch in range(num_epochs):
    # 1. Collect trajectories for each policy block
    all_data = []
    for i in range(M):
        block_idxs = list(range(i * num_envs // M, (i + 1) * num_envs // M))
        block_envs = [envs[j] for j in block_idxs]
        data = collect_trajectories(agent, block_envs, horizon, policy_idx=i)
        all_data.append(data)

    # 2. Update agent
    agent.update(all_data)

    # 3. Logging
    avg_ret = np.mean([np.sum(d["rewards"]) for d in all_data])
    print(f"Epoch {epoch+1}/{num_epochs} | Avg Return per env: {avg_ret:.3f}")

# Evaluation
eval_returns = []
for _ in range(eval_episodes):
    obs, _ = envs[0].reset(seed=SEED)
    done = False
    ep_ret = 0.0
    while not done:
        action, _, _ = agent.act(torch.tensor(obs, dtype=torch.float32), policy_idx=0, deterministic=True)
        obs, reward, terminated, truncated, _ = envs[0].step(action.cpu().numpy())
        done = terminated or truncated
        ep_ret += reward
    eval_returns.append(ep_ret)

avg_eval_ret = np.mean(eval_returns)
print(f"Average Return over {eval_episodes} eval episodes: {avg_eval_ret:.3f}")

# Save results
with open("logs/final_results.txt", "w") as f:
    f.write(f"Average Return: {avg_eval_ret:.3f}\n")

# Save best policy
torch.save(agent.get_leader_state_dict(), "logs/best_policy.pt")