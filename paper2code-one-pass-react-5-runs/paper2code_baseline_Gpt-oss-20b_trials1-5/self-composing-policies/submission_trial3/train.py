#!/usr/bin/env python3
"""
Training driver for all benchmarks.

Usage:
    python train.py --benchmark meta-world
"""

import argparse
import json
import os
import random
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from compo_net import CompoNet
from utils import make_env, load_sequences, save_curve
from tqdm import trange

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

def build_policy(env, algo, hidden_dim, encoder_type):
    """
    Build a policy network as used by stable‑baselines3.

    For SAC we return a deterministic policy (mean only).
    For PPO we return a stochastic policy (logits).
    """
    if encoder_type == "mlp":
        input_dim = env.observation_space.shape[0]
    else:
        # Atari: (3,210,160)
        input_dim = 3 * 210 * 160
    action_dim = env.action_space.n if isinstance(env.action_space, gym.spaces.Discrete) \
                 else env.action_space.shape[0]
    net = CompoNet(input_dim, action_dim, hidden_dim, encoder_type)
    return net

def run_training(env_name, algo, hidden_dim, timesteps, encoder_type):
    env = DummyVecEnv([lambda: make_env(env_name)])
    policy_net = build_policy(env.envs[0], algo, hidden_dim, encoder_type)
    policy_net.add_module()  # first module

    if algo == "ppo":
        model = PPO(policy_net, env, verbose=0, tensorboard_log=None,
                    n_steps=128, batch_size=256, learning_rate=2.5e-4,
                    clip_range=0.2, ent_coef=0.01, max_grad_norm=0.5,
                    gae_lambda=0.95)
    else:
        model = SAC(policy_net, env, verbose=0, tensorboard_log=None,
                    learning_rate=1e-3, buffer_size=1_000_000,
                    learning_starts=5_000, batch_size=128,
                    tau=0.005, gamma=0.99, train_freq=1,
                    gradient_steps=1)
    # Train
    model.learn(total_timesteps=timesteps)

    # Evaluate success curve
    success_curve = []
    for _ in trange(10, desc="Eval"):
        obs = env.reset()[0]
        done = False
        episode_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward[0]
        success_curve.append(episode_reward)
    return np.array(success_curve)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True,
                        choices=["meta-world", "spaceinvaders", "freeway"])
    args = parser.parse_args()
    seq_cfg = load_sequences()[args.benchmark]

    env_prefix = seq_cfg["env_prefix"]
    tasks = seq_cfg["tasks"] * seq_cfg["repeat"]
    algo = seq_cfg["algo"]
    hidden_dim = seq_cfg["hidden_dim"]
    timesteps = seq_cfg["timesteps"]

    # Determine encoder type
    encoder_type = "mlp" if "MetaWorld" in env_prefix else "cnn"

    results_dir = Path("results") / args.benchmark
    os.makedirs(results_dir, exist_ok=True)

    # Train baseline (random policy) for comparison
    baseline_curve = []
    for _ in trange(10, desc="Baseline"):
        env = DummyVecEnv([lambda: make_env(env_prefix, task_id=tasks[0])])
        obs = env.reset()[0]
        done = False
        ep_r = 0.0
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_r += reward[0]
        baseline_curve.append(ep_r)

    # Train CompoNet
    compo_curve = np.zeros(len(tasks))
    for idx, task in enumerate(tqdm(tasks, desc="CompoNet")):
        env_name = env_prefix if "MetaWorld" in env_prefix else f"{env_prefix}"
        # For Atari we need to set mode
        if "ALE" in env_prefix:
            env_name = f"{env_prefix}"
            env = DummyVecEnv([lambda: make_env(env_name, task_id=task)])
        else:
            env = DummyVecEnv([lambda: make_env(env_name)])
        # Add module
        # (we reuse the same policy network across tasks – the
        #  CompoNet instance keeps adding modules internally)
        # Here we just trigger the learning for the current task
        policy_net = build_policy(env.envs[0], algo, hidden_dim, encoder_type)
        policy_net.add_module()  # new module for this task

        if algo == "ppo":
            model = PPO(policy_net, env, verbose=0, tensorboard_log=None,
                        n_steps=128, batch_size=256, learning_rate=2.5e-4,
                        clip_range=0.2, ent_coef=0.01, max_grad_norm=0.5,
                        gae_lambda=0.95)
        else:
            model = SAC(policy_net, env, verbose=0, tensorboard_log=None,
                        learning_rate=1e-3, buffer_size=1_000_000,
                        learning_starts=5_000, batch_size=128,
                        tau=0.005, gamma=0.99, train_freq=1,
                        gradient_steps=1)
        model.learn(total_timesteps=timesteps)
        # Evaluate
        obs = env.reset()[0]
        done = False
        ep_r = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_r += reward[0]
        compo_curve[idx] = ep_r

    # Save curves
    save_curve(results_dir / "baseline.npy", np.array(baseline_curve))
    save_curve(results_dir / "compo.npy", compo_curve)

    # Simple report
    print("\n=== Report ===")
    print(f"Benchmark: {args.benchmark}")
    print(f"Baseline mean reward: {baseline_curve.mean():.2f}")
    print(f"CompoNet mean reward: {compo_curve.mean():.2f}")

if __name__ == "__main__":
    main()