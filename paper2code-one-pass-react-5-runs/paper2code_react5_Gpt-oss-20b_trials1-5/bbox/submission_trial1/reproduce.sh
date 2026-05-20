#!/usr/bin/env bash
# reproduce.sh
set -euo pipefail

# Install dependencies
pip install --quiet -r requirements.txt

# Ensure we have a GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "No NVIDIA GPU found – training will run on CPU."
fi

# Train the adapter
python train.py \
    --base-model microsoft/deberta-v3-base \
    --k 3 \
    --max-length 256 \
    --temperature 0.7 \
    --epochs 3 \
    --lr 5e-6 \
    --train-fraction 10 \
    --val-fraction 20 \
    --output-dir ./checkpoints

# Evaluate the trained adapter
python evaluate.py \
    --checkpoint ./checkpoints/adapter.pt \
    --base-model microsoft/deberta-v3-base \
    --k 3