#!/usr/bin/env python3
"""
Simple REINFORCE training loop for CompoNet on CartPole-v1.

The script trains on a sequence of 2 tasks.  Each task is the
same environment but with a different reward scaling to simulate
a change in the task objective.  The agent learns incrementally,
adding a new CompoModule for each task and freezing the
previous ones.  After training, the script prints the final
average return on the last task to demonstrate knowledge transfer.
"""

import math
import random
import gymnasium as gym
import numpy as np
import torch
from torch import optim
from typing import List, Tuple

from src.compo import CompoNet, get_policy

# ----- Hyperparameters -----
NUM_TASKS = 2
EPISODES_PER_TASK = 10
MAX_STEPS = 200          # CartPole max steps
GAMMA = 0.99
LR = 3e-4
SEED = 42

# ----- Environment wrapper -----
class RewardScaledEnv(gym.Env):
    """Wraps a Gym environment and scales the reward by a given factor."""
    def __init__(self, env_name: str, scale: float = 1.0):
        self.env = gym.make(env_name)
        self.scale = scale

    def reset(self, **kwargs):
        obs, _ = self.env.reset(**kwargs)
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward *= self.scale
        return np.array(obs, dtype=np.float32), reward, terminated, truncated, info

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space


# ----- Utility functions -----
def discounted_returns(rewards: List[float], gamma: float) -> List[float]:
    """Compute discounted returns for a list of rewards."""
    R = 0.0
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns


def train_task(net: CompoNet,
               env: gym.Env,
               optimizer: optim.Optimizer,
               device: torch.device):
    """Train the current task for a few episodes using REINFORCE."""
    net.train()
    all_returns = []
    for ep in range(EPISODES_PER_TASK):
        state = env.reset()
        log_probs: List[float] = []
        rewards: List[float] = []

        for t in range(MAX_STEPS):
            action, logp = get_policy(net, device)(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            log_probs.append(logp)
            rewards.append(reward)
            state = next_state
            if terminated or truncated:
                break

        returns = discounted_returns(rewards, GAMMA)
        loss = 0.0
        for logp, R in zip(log_probs, returns):
            loss -= logp * R
        loss /= len(log_probs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        episode_return = sum(rewards)
        all_returns.append(episode_return)

    avg_return = sum(all_returns) / len(all_returns)
    return avg_return


def evaluate(net: CompoNet, env: gym.Env, n_episodes: int = 10) -> float:
    """Evaluate the current policy (last module only)."""
    net.eval()
    total = 0.0
    for _ in range(n_episodes):
        state = env.reset()
        ep_ret = 0.0
        for _ in range(MAX_STEPS):
            action, _ = get_policy(net, torch.device('cpu'))(state)
            state, reward, terminated, truncated, _ = env.step(action)
            ep_ret += reward
            if terminated or truncated:
                break
        total += ep_ret
    return total / n_episodes


# ----- Main training loop -----
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create the base environment
    env_name = 'CartPole-v1'
    base_env = gym.make(env_name)

    # Compute state and action dimensions
    state_dim = base_env.observation_space.shape[0]
    action_dim = base_env.action_space.n

    # Initialize CompoNet
    net = CompoNet(state_dim, action_dim).to(device)

    # Optimizer (only parameters of the last module are trainable)
    optimizer = optim.Adam(net.parameters(), lr=LR)

    task_returns: List[float] = []

    for t in range(NUM_TASKS):
        print(f"\n=== Training Task {t} ===")
        net.add_module()  # new module for this task

        # Scale reward to simulate a different task objective
        reward_scale = 1.0 + 0.5 * t
        env = RewardScaledEnv(env_name, scale=reward_scale)

        avg_ret = train_task(net, env, optimizer, device)
        task_returns.append(avg_ret)
        print(f"Task {t}: avg return {avg_ret:.2f}")

    # Final evaluation on the last task
    final_ret = evaluate(net, env)
    print(f"\nFinal evaluation on Task {NUM_TASKS - 1}: avg return {final_ret:.2f}")

    # Save a minimal checkpoint (only the last module)
    torch.save(net.modules[-1].state_dict(), 'final_module.pt')
    print("\nTraining finished.  Final module saved to 'final_module.pt'.")


if __name__ == "__main__":
    main()