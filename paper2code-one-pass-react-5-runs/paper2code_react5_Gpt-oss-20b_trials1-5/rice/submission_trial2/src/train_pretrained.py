import gymnasium as gym
import torch
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from src.utils import set_seed, get_action_space_dims

set_seed(42)
env_id = "Hopper-v3"
env = make_vec_env(env_id, n_envs=1, seed=0)
obs_dim, act_dim = get_action_space_dims(env)

model = PPO(
    "MlpPolicy",
    env,
    verbose=0,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
    device="cpu"
)

timesteps = 200_000
model.learn(total_timesteps=timesteps, log_interval=10)
model.save("pretrained.zip")

# Evaluate
mean_reward = 0
for _ in range(10):
    obs, _ = env.reset()
    done = False
    total = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total += reward
    mean_reward += total
print(f"Pre‑trained PPO finished: mean reward = {mean_reward/10:.2f}")