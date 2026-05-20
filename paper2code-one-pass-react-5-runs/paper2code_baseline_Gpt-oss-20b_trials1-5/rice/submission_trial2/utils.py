import numpy as np
import torch
import random

def set_global_seeds(seed: int = 42):
    """Set seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def evaluate_policy(env, model, n_episodes=10, seed=42):
    """Run evaluation episodes and return average reward."""
    env.seed(seed)
    obs = env.reset()
    total_rewards = []
    for _ in range(n_episodes):
        obs = env.reset()
        episode_reward = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
        total_rewards.append(episode_reward)
    return np.mean(total_rewards)