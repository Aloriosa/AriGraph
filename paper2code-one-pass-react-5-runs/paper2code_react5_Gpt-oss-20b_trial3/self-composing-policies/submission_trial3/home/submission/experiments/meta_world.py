#!/usr/bin/env python
"""
Training script for the Meta‑World continuous‑control benchmark.
Trains 20 tasks (10 different, each repeated twice) with SAC + CompoNet.
"""

import os
import gymnasium as gym
import metaworld
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from componet.policy import CompoNetActorCriticSAC
from utils.metrics import evaluate_policy
from tqdm import tqdm

# ------------- Configuration ---------------------------------------
TASKS = [
    "hammer-v2",
    "push-wall-v2",
    "faucet-close-v2",
    "push-back-v2",
    "stick-pull-v2",
    "handle-press-side-v2",
    "push-v2",
    "shelf-place-v2",
    "window-close-v2",
    "peg-unplug-side-v2",
]
# repeat twice -> 20 tasks
TASKS = TASKS * 2

TIMESTEPS_PER_TASK = 1_000_000
N_EVAL_EPISODES = 10
SEED = 42

# ------------------------------------------------------------------
# 1. Create environment list
# ------------------------------------------------------------------
envs = []
for task_id in TASKS:
    env = gym.make(task_id, render_mode=None)
    envs.append(env)

# ------------------------------------------------------------------
# 2. Train sequentially
# ------------------------------------------------------------------
results = []

for i, env in enumerate(tqdm(envs, desc="Meta‑World tasks")):
    torch.manual_seed(SEED + i)
    env.seed(SEED + i)

    # Wrap in DummyVecEnv to satisfy SB3
    vec_env = DummyVecEnv([lambda: env])

    # Instantiate model with custom policy
    model = SAC(
        policy=CompoNetActorCriticSAC,
        env=vec_env,
        learning_rate=3e-4,
        batch_size=128,
        buffer_size=1_000_000,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef=0.2,
        verbose=0,
        policy_kwargs=dict(
            features_extractor_class=None,
            features_extractor_kwargs=dict(),
            net_arch=[256, 256],
        ),
    )

    # Add new module for this task
    observation_space = env.observation_space
    action_space = env.action_space
    if isinstance(action_space, gym.spaces.Box):
        action_dim = action_space.shape[0]
    else:
        raise NotImplementedError("Non‑continuous actions not supported in Meta‑World")

    state_dim = observation_space.shape[0]
    model.policy.add_module(state_dim, action_dim)

    # Train
    model.learn(total_timesteps=TIMESTEPS_PER_TASK, reset_num_timesteps=True)

    # Evaluate
    mean_ret, _ = evaluate_policy(
        env, lambda obs: model.predict(obs, deterministic=True)[0], n_episodes=N_EVAL_EPISODES
    )

    results.append(
        {
            "method": "CompoNet",
            "task": f"{i}-{task_id}",
            "env": "MetaWorld",
            "mean_return": mean_ret,
            "success_rate": 0.0,  # not defined for continuous tasks
        }
    )

    # Save checkpoint locally (not committed)
    os.makedirs("tmp", exist_ok=True)
    model.save(f"tmp/componet_meta_{i}")

# ------------------------------------------------------------------
# 3. Write CSV
# ------------------------------------------------------------------
import csv

with open("meta_world_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["method", "task", "env", "mean_return", "success_rate"])
    writer.writeheader()
    writer.writerows(results)

print("Meta‑World training finished – results written to meta_world_results.csv")