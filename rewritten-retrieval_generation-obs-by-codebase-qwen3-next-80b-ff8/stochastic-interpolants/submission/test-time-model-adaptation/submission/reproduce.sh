#!/bin/bash

# Set up environment
set -e  # Exit on any error

# Install required packages
apt-get update && apt-get install -y python3 python3-pip python3-dev

# Install Python dependencies
pip3 install torch torchvision timm numpy cma tqdm scipy scikit-learn

# Create output directory
mkdir -p results

# Run FOA reproduction script
python3 main.py \
    --model vit_base_patch16_224 \
    --dataset imagenet_c \
    --quantization_bit 8 \
    --batch_size 64 \
    --fitness_lambda 0.4 \
    --population_size 27 \
    --max_iterations 10 \
    --output results/results.csv

echo "Reproduction completed. Results saved to results/results.csv"