#!/usr/bin/env python3
"""
Zero‑shot evaluation of the trained FRE policy on a new reward function.
The evaluation reward is a simple “pole‑angle” reward (higher absolute angle → higher reward).
"""
import pickle
import os
import numpy as np
import torch
import gymnasium as gym
from models import RewardEncoder, QNetwork
from utils import OfflineDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENCODER_FILE = "encoder.pt"
QFILE = "q.pt"
RESULT_FILE = "results.txt"

def load_models():
    encoder = RewardEncoder(state_dim=4).to(DEVICE)
    qnet = QNetwork(state_dim=4, latent_dim=32).to(DEVICE)
    encoder.load_state_dict(torch.load(ENCODER_FILE, map_location=DEVICE))
    qnet.load_state_dict(torch.load(QFILE, map_location=DEVICE))
    encoder.eval()
    qnet.eval()
    return encoder, qnet

def downstream_reward(state: np.ndarray) -> float:
    """
    Example downstream reward: maximize absolute pole angle (state[2]).
    """
    return abs(state[2])  # larger angle → higher reward

def main():
    encoder, qnet = load_models()

    # Sample 8 encoder points from the offline dataset
    with open("offline_data.pkl", "rb") as f:
        transitions = pickle.load(f)
    indices = np.random.choice(len(transitions), size=8, replace=False)
    enc_states = torch.tensor(
        [transitions[i]["state"] for i in indices], dtype=torch.float32
    ).unsqueeze(0).to(DEVICE)  # [1, K, state_dim]

    # Compute rewards for these states under the downstream function
    rewards = torch.tensor(
        [downstream_reward(transitions[i]["state"]) for i in indices],
        dtype=torch.float32,
    ).unsqueeze(0).unsqueeze(-1).to(DEVICE)  # [1, K, 1]

    # Encode to latent z
    with torch.no_grad():
        z, _, _ = encoder(enc_states, rewards)

    # Run policy for a few episodes
    env = gym.make("CartPole-v1", render_mode=None)
    episode_returns = []
    for ep in range(20):
        obs, _ = env.reset(seed=ep)
        total = 0.0
        for t in range(200):
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                q_vals = qnet(obs_t, z).squeeze(0)  # [n_actions]
                action = q_vals.argmax().item()
            obs, _, terminated, truncated, _ = env.step(action)
            total += downstream_reward(obs)  # use downstream reward for return
            if terminated or truncated:
                break
        episode_returns.append(total)
    env.close()

    mean_ret = np.mean(episode_returns)
    std_ret = np.std(episode_returns)
    with open(RESULT_FILE, "w") as f:
        f.write(f"Zero‑shot evaluation on downstream reward (pole angle)\n")
        f.write(f"Mean return over 20 episodes: {mean_ret:.2f} ± {std_ret:.2f}\n")
    print(f"Evaluation finished – results written to {RESULT_FILE}")

if __name__ == "__main__":
    main()