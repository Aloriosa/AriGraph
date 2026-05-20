#!/usr/bin/env python3
import os
import json
import random
import numpy as np
import torch
import gymnasium as gym
from tqdm import trange
from compo_net import CompoNet

# --------------------------------------------------------------------------- #
# Utility helpers
# --------------------------------------------------------------------------- #
def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# --------------------------------------------------------------------------- #
# Training a single task
# --------------------------------------------------------------------------- #
def train_task(compo: CompoNet,
               env: gym.Env,
               task_idx: int,
               num_episodes: int = 200,
               gamma: float = 0.99,
               lr: float = 1e-3) -> dict:
    """
    Train the newly added module on the given task.
    Returns a dict with simple metrics.
    """
    # Add a new module for this task
    compo.add_task()
    new_module = compo.modules_list[-1]
    optimizer = torch.optim.Adam(new_module.parameters(), lr=lr)

    episode_lengths = []
    episode_returns = []

    for ep in trange(num_episodes, desc=f"Task {task_idx}"):
        # Reset environment
        obs, _ = env.reset(seed=42 + ep + task_idx * 1000)
        log_probs = []
        rewards = []
        done = False
        step = 0

        while not done:
            state_tensor = torch.tensor(obs,
                                        dtype=torch.float32).unsqueeze(0)
            probs = compo(state_tensor)  # (1, A)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            log_probs.append(log_prob)
            rewards.append(reward)
            step += 1

        # Compute discounted returns
        R = 0.0
        returns = []
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)

        # Policy gradient loss (REINFORCE)
        log_probs = torch.stack(log_probs)
        loss = -torch.mean(log_probs * returns)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        episode_lengths.append(step)
        episode_returns.append(sum(rewards))

    # Evaluation
    eval_success = 0
    eval_episodes = 10
    for _ in range(eval_episodes):
        obs, _ = env.reset()
        done = False
        step = 0
        while not done:
            state_tensor = torch.tensor(obs,
                                        dtype=torch.float32).unsqueeze(0)
            probs = compo(state_tensor)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            obs, _, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            step += 1
        if step >= 500:  # success threshold for CartPole
            eval_success += 1

    metrics = {
        "task": task_idx,
        "episode_length_mean": np.mean(episode_lengths),
        "episode_return_mean": np.mean(episode_returns),
        "success_rate": eval_success / eval_episodes
    }
    return metrics

# --------------------------------------------------------------------------- #
# Main training loop over a sequence of tasks
# --------------------------------------------------------------------------- #
def main():
    set_global_seed(42)

    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    compo = CompoNet(state_dim=state_dim,
                     action_dim=action_dim,
                     hidden_dim=128)

    all_metrics = []
    num_tasks = 2  # Two‑task sequence (same env, different seeds)

    for task_idx in range(num_tasks):
        metrics = train_task(compo,
                             env,
                             task_idx,
                             num_episodes=200,
                             gamma=0.99,
                             lr=1e-3)
        all_metrics.append(metrics)
        print(f"Task {task_idx} metrics: {metrics}")

    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    print("Training finished. Metrics saved to results/metrics.json")


if __name__ == "__main__":
    main()