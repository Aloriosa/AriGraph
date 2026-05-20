#!/usr/bin/env python3
import os
import gym
import numpy as np
from stable_baselines3 import PPO

SEED = 42
np.random.seed(SEED)

ENV_ID = "CartPole-v1"
PRETRAINED_MODEL = "../models/pretrained_ppo.zip"
REFINED_MODEL = "../models/refined_ppo.zip"
N_EPISODES = 20

def evaluate_policy(model_path):
    model = PPO.load(model_path, verbose=0)
    env = gym.make(ENV_ID)
    env.seed(SEED)
    returns = []
    for _ in range(N_EPISODES):
        obs = env.reset()
        done = False
        total = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            total += reward
        returns.append(total)
    return np.mean(returns), np.std(returns)

if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    pre_ret, pre_std = evaluate_policy(PRETRAINED_MODEL)
    ref_ret, ref_std = evaluate_policy(REFINED_MODEL)

    with open("logs/results.txt", "w") as f:
        f.write(f"Pre‑trained policy average return: {pre_ret:.2f} ± {pre_std:.2f}\n")
        f.write(f"Refined policy average return: {ref_ret:.2f} ± {ref_std:.2f}\n")
        f.write(f"Improvement factor: {ref_ret/pre_ret:.2f}×\n")

    print("Evaluation results written to logs/results.txt")