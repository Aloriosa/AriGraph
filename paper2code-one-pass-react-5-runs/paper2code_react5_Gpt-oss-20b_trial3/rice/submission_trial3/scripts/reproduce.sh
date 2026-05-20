#!/usr/bin/env bash
set -euo pipefail

# Install dependencies
pip install -r requirements.txt

# Create output directory
mkdir -p logs

# Train a pre‑trained policy (10k steps for demo purposes)
python - <<'PY'
import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

env_name="CartPole-v1"
env = DummyVecEnv([lambda: gym.make(env_name)])
model = PPO("MlpPolicy", env, learning_rate=3e-4, batch_size=64, verbose=0)
model.learn(total_timesteps=10000)
model.save("pretrained_cartpole.zip")
print("Pre‑trained policy saved.")
PY

# Run RICE refinement
python - <<'PY'
import os
from src.rice import RICE

env_name = "CartPole-v1"
rice = RICE(env_name, pretrained_path="pretrained_cartpole.zip", seed=42)
print("Training mask network...")
rice.train_mask(epochs=3, samples_per_epoch=200)
print("Refining policy...")
rice.refine(iter_steps=200, num_iters=50)
print("Evaluating refined policy...")
avg_reward = rice.evaluate(episodes=20)
print(f"Average reward after refinement: {avg_reward:.2f}")
PY

echo "Reproduction finished."