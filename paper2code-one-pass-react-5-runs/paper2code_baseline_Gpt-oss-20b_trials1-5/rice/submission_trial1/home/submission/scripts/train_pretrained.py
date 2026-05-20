#!/usr/bin/env python3
import os
import gym
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

ENV_ID = "CartPole-v1"
TIMESTEPS = 200_000
MODEL_DIR = "../models/pretrained_ppo.zip"

if __name__ == "__main__":
    # Create vectorized environment
    env = make_vec_env(ENV_ID, n_envs=4, seed=SEED)

    # PPO hyper‑parameters are chosen for quick training
    model = PPO("MlpPolicy",
                env,
                verbose=1,
                n_steps=2048,
                batch_size=64,
                learning_rate=3e-4,
                gamma=0.99,
                seed=SEED)

    print("Training pre‑trained policy...")
    model.learn(total_timesteps=TIMESTEPS)

    # Save the model
    os.makedirs(os.path.dirname(MODEL_DIR), exist_ok=True)
    model.save(MODEL_DIR)
    print(f"Pre‑trained policy saved to {MODEL_DIR}")