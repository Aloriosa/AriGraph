#!/usr/bin/env bash
set -euo pipefail

# Update and install system packages
apt-get update -qq
apt-get install -y -qq python3 python3-pip git

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -q -r requirements.txt

# Download the synthetic offline dataset (generated on the fly)
echo "Generating synthetic offline dataset for CartPole..."
python - <<'PY'
import gym
import numpy as np
import pickle
import os

env_name = 'CartPole-v1'
env = gym.make(env_name)
max_steps = 200  # typical horizon

# Generate 10000 trajectories with a random policy
dataset = []
for _ in range(10000):
    obs = env.reset(seed=42)
    for t in range(max_steps):
        action = env.action_space.sample()
        next_obs, reward, done, _ = env.step(action)
        dataset.append((obs, action, next_obs, reward))
        obs = next_obs
        if done:
            break

# Save the dataset
os.makedirs('data', exist_ok=True)
with open('data/cartpole_offline.pkl', 'wb') as f:
    pickle.dump(dataset, f)
print("Dataset saved to data/cartpole_offline.pkl")
PY

# Run training and evaluation
echo "Running FRE training and evaluation..."
python src/train.py

echo "Reproduction complete."
echo "Metrics saved to metrics.csv"