import gymnasium as gym
import numpy as np
import torch
import pickle
import os
from pathlib import Path

def generate_offline_dataset(env_name: str, num_episodes: int = 50,
                             max_steps_per_ep: int = 200,
                             seed: int = 0,
                             save_path: str = "offline_dataset.pt"):
    """
    Generates a small offline dataset by rolling out a random policy.
    """
    rng = np.random.RandomState(seed)
    env = gym.make(env_name, render_mode=None, disable_env_checker=True)
    env.action_space.seed(seed)
    dataset = []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed + ep)
        for t in range(max_steps_per_ep):
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, info = env.step(action)
            dataset.append((obs, action, reward, next_obs, terminated or truncated))
            if terminated or truncated:
                break
            obs = next_obs

    env.close()

    # Convert to numpy arrays
    obs = np.array([d[0] for d in dataset], dtype=np.float32)
    actions = np.array([d[1] for d in dataset], dtype=np.float32)
    rewards = np.array([d[2] for d in dataset], dtype=np.float32)
    next_obs = np.array([d[3] for d in dataset], dtype=np.float32)
    terminals = np.array([d[4] for d in dataset], dtype=bool)

    data = {
        "obs": obs,
        "actions": actions,
        "rewards": rewards,
        "next_obs": next_obs,
        "terminals": terminals
    }
    torch.save(data, save_path)
    print(f"Offline dataset written to {save_path} ({len(dataset)} transitions)")
    return data

def load_offline_dataset(path: str):
    return torch.load(path, map_location="cpu")