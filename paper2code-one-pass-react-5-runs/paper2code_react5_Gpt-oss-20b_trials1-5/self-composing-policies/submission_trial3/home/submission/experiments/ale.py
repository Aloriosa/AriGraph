#!/usr/bin/env python
"""
Training script for Atari benchmarks (SpaceInvaders and Freeway).
Uses PPO + CompoNet.
"""

import os
import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from componet.policy import CompoNetActorCriticPPO
from utils.metrics import evaluate_policy
from tqdm import tqdm

# ------------- Configuration ---------------------------------------
# SpaceInvaders modes 0‑9 (10 tasks)
# Freeway modes 0‑6 (7 tasks)
SPACEINVADERS_TASKS = [f"SpaceInvaders-v5" for _ in range(10)]
FREEWAY_TASKS = [f"Freeway-v5" for _ in range(7)]

ALL_TASKS = SPACEINVADERS_TASKS + FREEWAY_TASKS
ENV_NAMES = ["SpaceInvaders"] * 10 + ["Freeway"] * 7

TIMESTEPS_PER_TASK = 1_000_000
N_EVAL_EPISODES = 10
SEED = 43

# ------------------------------------------------------------------
# 1. Create environment list
# ------------------------------------------------------------------
envs = []
for env_name in ALL_TASKS:
    env = gym.make(env_name, render_mode=None)
    envs.append(env)

# ------------------------------------------------------------------
# 2. Train sequentially
# ------------------------------------------------------------------
results = []

for i, env in enumerate(tqdm(envs, desc="Atari tasks")):
    torch.manual_seed(SEED + i)
    env.seed(SEED + i)

    vec_env = DummyVecEnv([lambda: env])

    model = PPO(
        policy=CompoNetActorCriticPPO,
        env=vec_env,
        learning_rate=2.5e-4,
        n_steps=128,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        clip_range=0.2,
        verbose=0,
        policy_kwargs=dict(
            features_extractor_class=None,
            features_extractor_kwargs=dict(),
            net_arch=[512, 512],
        ),
    )

    # Add module
    observation_space = env.observation_space
    action_space = env.action_space
    if isinstance(action_space, gym.spaces.Discrete):
        action_dim = action_space.n
    else:
        raise NotImplementedError("Non‑discrete actions not supported in Atari")

    state_dim = observation_space.shape[0]
    model.policy.add_module(state_dim, action_dim)

    # Train
    model.learn(total_timesteps=TIMESTEPS_PER_TASK, reset_num_timesteps=True)

    # Evaluate
    mean_ret, success_rate = evaluate_policy(
        env,
        lambda obs: model.predict(obs, deterministic=True)[0],
        n_episodes=N_EVAL_EPISODES,
        success_threshold=0.9 * 100,  # rough threshold, not exact
    )

    results.append(
        {
            "method": "CompoNet",
            "task": f"{i}-{env_name}",
            "env": ENV_NAMES[i],
            "mean_return": mean_ret,
            "success_rate": success_rate,
        }
    )

    # Save checkpoint locally
    os.makedirs("tmp", exist_ok=True)
    model.save(f"tmp/componet_ale_{i}")

# ------------------------------------------------------------------
# 3. Write CSV
# ------------------------------------------------------------------
import csv

with open("ale_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["method", "task", "env", "mean_return", "success_rate"])
    writer.writeheader()
    writer.writerows(results)

print("Atari training finished – results written to ale_results.csv")