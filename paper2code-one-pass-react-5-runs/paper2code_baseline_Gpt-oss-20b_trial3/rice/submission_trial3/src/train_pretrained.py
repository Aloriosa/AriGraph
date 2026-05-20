import gymnasium as gym
import numpy as np
import torch
import os
import random
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from utils import set_global_seeds
from constants import *

def main():
    set_global_seeds(PRETRAIN_SEED)
    env = DummyVecEnv([lambda: gym.make(ENV_NAME)])
    model = PPO("MlpPolicy", env, verbose=1,
                tensorboard_log="./logs/pretrain/",
                learning_rate=3e-4,
                batch_size=64,
                n_steps=2048,
                gamma=0.99,
                seed=PRETRAIN_SEED)

    model.learn(total_timesteps=PRETRAIN_TIMESTEPS,
                log_interval=PRETRAIN_LOG_FREQ)
    os.makedirs("results", exist_ok=True)
    model.save("results/baseline.zip")
    print("\nBaseline model saved to results/baseline.zip")

if __name__ == "__main__":
    main()