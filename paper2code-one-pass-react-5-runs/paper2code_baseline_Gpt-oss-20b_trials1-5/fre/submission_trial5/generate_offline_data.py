#!/usr/bin/env python3
"""
Generate a small offline dataset for CartPole using a random policy.
The dataset is saved as 'offline_data.pkl'.
"""
import gymnasium as gym
import numpy as np
import pickle
import os

DATA_FILE = "offline_data.pkl"

def generate_dataset(env_name="CartPole-v1", n_episodes=200, max_steps=200):
    env = gym.make(env_name, render_mode=None)
    transitions = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        for t in range(max_steps):
            action = env.action_space.sample()
            next_obs, env_reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            transitions.append({
                "state": obs,
                "action": action,
                "next_state": next_obs,
                "env_reward": env_reward,
            })
            obs = next_obs
            if done:
                break

    env.close()
    return transitions

def main():
    if os.path.exists(DATA_FILE):
        print(f"{DATA_FILE} already exists – skipping generation.")
        return
    print("Generating dataset …")
    data = generate_dataset()
    with open(DATA_FILE, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved {len(data)} transitions to {DATA_FILE}.")

if __name__ == "__main__":
    main()