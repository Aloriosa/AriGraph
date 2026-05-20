#!/usr/bin/env bash
set -e

echo "Installing dependencies..."
pip install -q -U pip
pip install -q gymnasium[atari,box2d,classic_control,robotics] stable-baselines3 torch numpy tqdm

echo "Running reproduction script..."
python run.py