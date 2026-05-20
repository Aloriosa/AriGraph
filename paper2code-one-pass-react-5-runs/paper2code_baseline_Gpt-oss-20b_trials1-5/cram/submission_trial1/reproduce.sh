#!/usr/bin/env bash
set -euo pipefail

# Update package lists and install Python 3
apt-get update -y
apt-get install -y python3-pip

# Install required Python packages
pip install --upgrade pip
pip install transformers torch

# Create output directory
mkdir -p output

# Run compression
echo "=== Running compression ==="
python3 compress.py \
    --model_name gpt2 \
    --text_file sample_text.txt \
    --mem_tokens 1 \
    --max_steps 2000 \
    --learning_rate 1e-3 \
    --output_dir output

# Run decoding
echo "=== Running decoding ==="
python3 decode.py \
    --model_name gpt2 \
    --mem_file output/mem.pt \
    --mem_tokens 1 \
    --output_dir output \
    --text_file sample_text.txt

echo "=== Reproduction finished ==="