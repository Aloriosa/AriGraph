import numpy as np
import torch
import gymnasium as gym
import random

def set_global_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def evaluate_policy(model, env, n_episodes=10, seed=0):
    """
    Run the policy for a few episodes and return the average total reward.
    """
    set_global_seeds(seed)
    total = 0.0
    for _ in range(n_episodes):
        obs, _ = env.reset(seed=seed)
        done = False
        ep_ret = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_ret += reward
        total += ep_ret
    return total / n_episodes