#!/usr/bin/env python3
"""Main training script for the CompoNet continual RL demo."""

import csv
import os
import math
import torch
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
from env_variations import CartPoleGravity
from compo_net import CompoNet
from utils import compute_returns, evaluate_policy

# Set random seeds for reproducibility
torch.manual_seed(42)

# Task configuration – different gravity values
TASKS = [
    {"name": "Task 0", "gravity": 9.8},
    {"name": "Task 1", "gravity": 15.0},
    {"name": "Task 2", "gravity": 5.0},
]

NUM_EPISODES = 200          # per task
MAX_STEPS = 200             # per episode
GAMMA = 0.99
LEARNING_RATE = 1e-3
BATCH_SIZE = 1              # on-policy (REINFORCE) – one episode at a time
EVAL_EPISODES = 10

def train_one_task(policy, env, epochs=NUM_EPISODES, gamma=GAMMA, lr=LEARNING_RATE):
    """
    Train the most recent module of the policy on a single task using REINFORCE.

    Args:
        policy: CompoNet instance (new module already added and previous frozen).
        env: Environment for this task.
        epochs: Number of episodes to train.
        gamma: Discount factor.
        lr: Learning rate.
    """
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, policy.parameters()), lr=lr)

    for ep in range(epochs):
        obs, _ = env.reset()
        log_probs = []
        rewards = []

        for _ in range(MAX_STEPS):
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            logits = policy(obs_t)
            probs = F.softmax(logits, dim=-1)
            m = torch.distributions.Categorical(probs)
            action = m.sample()
            log_prob = m.log_prob(action)

            obs, reward, terminated, truncated, _ = env.step(action.item())
            log_probs.append(log_prob)
            rewards.append(reward)

            if terminated or truncated:
                break

        # Compute discounted returns
        returns = compute_returns(rewards, gamma)
        returns = torch.tensor(returns, dtype=torch.float32)

        log_probs = torch.stack(log_probs)
        loss = -torch.mean(log_probs * returns)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def main():
    # Observation and action space dimensions (CartPole-v1)
    dummy_env = gym.make("CartPole-v1")
    obs_dim = dummy_env.observation_space.shape[0]
    action_dim = dummy_env.action_space.n
    dummy_env.close()

    policy = CompoNet(obs_dim, action_dim)

    # Create CSV for results
    csv_path = os.path.join("results", "results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task", "gravity", "avg_return"])

    for task_id, task_cfg in enumerate(TASKS):
        print(f"\n=== Training on {task_cfg['name']} (gravity={task_cfg['gravity']}) ===")
        env = CartPoleGravity(gravity=task_cfg["gravity"])

        # Add a new module for this task
        policy.add_module()
        # Freeze previous modules
        policy.freeze_modules()

        # Train the new module
        train_one_task(policy, env)

        # Evaluate
        avg_ret = evaluate_policy(policy, env, episodes=EVAL_EPISODES)
        print(f"Average return after training: {avg_ret:.2f}")

        # Append to CSV
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([task_id, task_cfg["gravity"], f"{avg_ret:.2f}"])

        env.close()

    print("\nAll tasks completed. Results written to:", csv_path)


if __name__ == "__main__":
    main()