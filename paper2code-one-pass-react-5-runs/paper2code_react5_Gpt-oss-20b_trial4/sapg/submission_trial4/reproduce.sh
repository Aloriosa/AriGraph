#!/usr/bin/env bash
set -e

# Install dependencies
pip install -q -r requirements.txt

# Run training
python sapg.py --env CartPole-v1 --num-envs 32 --num-policies 3 --total-steps 5000 --seed 42

echo "Reproduction finished. Results are in results.json"