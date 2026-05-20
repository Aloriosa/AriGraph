#!/usr/bin/env bash
set -euo pipefail

# Install python3 and pip
apt-get update
apt-get install -y python3 python3-pip

# Install python dependencies
pip install --no-cache-dir -r requirements.txt

# Create output directory
mkdir -p output

# Run the compression experiment
python3 compress_mem.py \
    --model_name EleutherAI/gpt-neo-125M \
    --text_file data/sample_text.txt \
    --mem_tokens 1 \
    --learning_rate 1e-3 \
    --num_steps 5000 \
    --output_dir output