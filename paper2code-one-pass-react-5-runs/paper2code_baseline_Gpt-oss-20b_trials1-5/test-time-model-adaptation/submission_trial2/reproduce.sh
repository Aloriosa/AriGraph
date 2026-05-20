#!/usr/bin/env bash
set -e

# Install dependencies
pip install -q -r requirements.txt

# Clone ImageNet‑C (if not already present)
if [ ! -d "imagenet-c" ]; then
  echo "Cloning ImageNet‑C dataset (≈ 1 GB)..."
  git clone https://github.com/hendrycks/imagenet-c.git imagenet-c
fi

# Run the experiment
echo "Running FOA on ImageNet‑C (severity 5)..."
python src/main.py \
    --dataset imagenet-c/imagenet_c/imagenet_c_val \
    --batch-size 64 \
    --num-prompt 3 \
    --lambda 0.4 \
    --num-generations 1 \
    --device cuda:0