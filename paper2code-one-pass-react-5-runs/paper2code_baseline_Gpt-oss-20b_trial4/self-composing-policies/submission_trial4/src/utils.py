"""Utility functions for training and evaluation."""

import math
import torch


def compute_returns(rewards, gamma: float = 0.99):
    """
    Compute discounted returns from a list of rewards.

    Args:
        rewards: List[float] of rewards in a single episode.
        gamma: Discount factor.
    Returns:
        List[float] of discounted returns for each timestep.
    """
    returns = []
    R = 0.0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    # Normalize returns for stability
    returns = torch.tensor(returns, dtype=torch.float32)
    if returns.std() > 0:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns.tolist()


def evaluate_policy(policy, env, episodes: int = 10, max_steps: int = 200):
    """
    Run a fixed number of episodes and return the average total reward.

    Args:
        policy: The policy network (an instance of CompoNet).
        env: The environment to evaluate on.
        episodes: Number of episodes to roll out.
        max_steps: Max steps per episode.
    Returns:
        Average total reward over all episodes.
    """
    total_reward = 0.0
    policy.eval()
    with torch.no_grad():
        for _ in range(episodes):
            obs, _ = env.reset()
            ep_reward = 0.0
            for _ in range(max_steps):
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                logits = policy(obs_t)
                probs = F.softmax(logits, dim=-1)
                m = torch.distributions.Categorical(probs)
                action = m.sample().item()
                obs, reward, terminated, truncated, _ = env.step(action)
                ep_reward += reward
                if terminated or truncated:
                    break
            total_reward += ep_reward
    policy.train()
    return total_reward / episodes