#!/usr/bin/env bash
set -e

# Install Python packages
python3 -m pip install --upgrade pip
pip install -r requirements.txt

# Train the baseline PPO agent
python3 -m src.train_pretrained

# Train the refined agent using RICE
python3 -m src.train_rice

# Evaluate both agents
python3 -m src.evaluate