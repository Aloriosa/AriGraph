#!/usr/bin/env bash
# Reproduce the memory‑token compression demo.
set -euo pipefail

# Update package list and install python3
apt-get update -y
apt-get install -y python3 python3-pip git

# Create a virtual environment (optional but recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install required Python packages
pip install --upgrade pip
pip install -r requirements.txt

# Run the compression demo script
python compress_mem.py \
    --model_name gpt2 \
    --text "The quick brown fox jumps over the lazy dog. It was a sunny day and the sky was clear." \
    --output results.json \
    --max_steps 2000 \
    --learning_rate 1e-3 \
    --seed 42

echo "Reproduction finished. Generated results are in results.json"