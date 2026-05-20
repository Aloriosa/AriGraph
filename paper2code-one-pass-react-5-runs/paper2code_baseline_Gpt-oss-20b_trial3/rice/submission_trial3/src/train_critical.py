import gymnasium as gym
import numpy as np
import os
import random
from stable_baselines3 import PPO
from utils import set_global_seeds
from constants import *

def compute_advantages(obs, reward, next_obs, done, value_fn, gamma=0.99):
    """
    Simple advantage estimate: A = r + gamma * V(next) - V(current)
    """
    v_current = value_fn(obs)
    v_next = value_fn(next_obs) if not done else 0.0
    return reward + gamma * v_next - v_current

def main():
    set_global_seeds(CRIT_SEED)
    env = gym.make(ENV_NAME)
    model = PPO.load("results/baseline.zip", env=env, verbose=0)
    model.set_env(env)  # ensure policy uses the same env

    all_states = []
    all_advantages = []

    for ep in range(CRIT_STEPS):
        obs, _ = env.reset(seed=CRIT_SEED + ep)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            next_obs, reward, done, truncated, info = env.step(action)
            adv = compute_advantages(obs, reward, next_obs, done, model.policy.value)
            all_states.append(obs.copy())
            all_advantages.append(adv)
            obs = next_obs

    all_advantages = np.array(all_advantages)
    threshold = np.percentile(all_advantages, 100 * (1 - CRIT_TOP_PERCENT))
    critical_states = np.array(all_states)[all_advantages >= threshold]
    np.save("results/critical_states.npy", critical_states)
    print(f"\nExtracted {len(critical_states)} critical states and saved to results/critical_states.npy")

if __name__ == "__main__":
    main()