# src/train_componet.py
"""
Continual learning training script that demonstrates the CompoNet
architecture on a short sequence of Gym environments.
The script trains each task sequentially, freezing all older modules,
and evaluates the final policy on each task.
"""

import math
import json
import os
import random
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from composenet_policy import CompoNet

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
TASKS = [
    "CartPole-v1",       # 4‑dim state, 2 actions
    "Acrobot-v1",        # 6‑dim state, 3 actions
    "MountainCar-v0",    # 2‑dim state, 3 actions
]
NUM_EPISODES_PER_TASK = 200
MAX_STEPS_PER_EPISODE = 500
GAMMA = 0.99
LR = 1e-3
DEVICE = torch.device("cpu")  # change to "cuda" if GPU is available

# --------------------------------------------------------------------------- #
# Utility functions
# --------------------------------------------------------------------------- #
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def evaluate_policy(env, policy, n_episodes=5):
    """Return the mean return over n_episodes."""
    returns = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        total_r = 0.0
        done = False
        t = 0
        while not done and t < MAX_STEPS_PER_EPISODE:
            state = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
            logits = policy(state)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample().item()
            obs, reward, done, _, _ = env.step(action)
            total_r += reward
            t += 1
        returns.append(total_r)
    return np.mean(returns)

# --------------------------------------------------------------------------- #
# Main training loop
# --------------------------------------------------------------------------- #
def main():
    set_seed(42)
    results = {}

    # Assume all tasks share the same state_dim and action_dim for simplicity
    # (they do not in reality, but this demo keeps the implementation simple)
    # We will instantiate the network with the state/action dimension of the
    # first environment; later tasks will be reshaped accordingly.
    first_env = gym.make(TASKS[0])
    state_dim = first_env.observation_space.shape[0]
    action_dim = first_env.action_space.n

    composenet = CompoNet(state_dim, action_dim)

    for task_idx, env_id in enumerate(TASKS):
        print(f"\n=== Training on task {task_idx+1}/{len(TASKS)}: {env_id} ===")
        env = gym.make(env_id)
        env.reset(seed=42)

        # Adapt network to the new environment if dimensions differ
        new_state_dim = env.observation_space.shape[0]
        new_action_dim = env.action_space.n
        if new_state_dim != state_dim or new_action_dim != action_dim:
            # Re‑initialize the network with the correct dimensions
            composenet = CompoNet(new_state_dim, new_action_dim)
            state_dim, action_dim = new_state_dim, new_action_dim

        # Add a new module
        if task_idx == 0:
            composenet.add_module()
        else:
            composenet.add_module()
            # Freeze all older modules
            for m in composenet.modules[:-1]:
                for p in m.parameters():
                    p.requires_grad = False

        # Optimizer only for the newest module
        optimizer = optim.Adam(composenet.modules[-1].parameters(), lr=LR)

        # Training loop
        for episode in tqdm(range(NUM_EPISODES_PER_TASK), desc="Episodes"):
            obs, _ = env.reset()
            log_probs = []
            rewards = []

            for t in range(MAX_STEPS_PER_EPISODE):
                state = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
                logits = composenet(state)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)

                next_obs, reward, done, _, _ = env.step(action.item())

                log_probs.append(log_prob)
                rewards.append(reward)

                obs = next_obs
                if done:
                    break

            # Compute discounted returns
            returns = []
            R = 0.0
            for r in reversed(rewards):
                R = r + GAMMA * R
                returns.insert(0, R)
            returns = torch.tensor(returns, dtype=torch.float32, device=DEVICE)

            # Normalize returns
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)

            # Policy gradient loss
            loss = 0.0
            for logp, R in zip(log_probs, returns):
                loss -= logp * R
            loss = loss / len(log_probs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluation
        avg_ret = evaluate_policy(env, composenet, n_episodes=5)
        print(f"Average return on {env_id}: {avg_ret:.2f}")
        results[env_id] = float(avg_ret)
        env.close()

    # Save results
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Final results ===")
    for env_id, ret in results.items():
        print(f"{env_id}: {ret:.2f}")

    # Simple overall summary
    mean_ret = np.mean(list(results.values()))
    print(f"\nMean return over all tasks: {mean_ret:.2f}")


if __name__ == "__main__":
    main()