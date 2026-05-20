#!/bin/bash

# Set up environment
apt-get update && apt-get install -y python3 python3-pip python3-torchvision python3-torch

# Install required packages
pip3 install torch torchvision tqdm matplotlib numpy scikit-learn

# Create directory structure
mkdir -p /home/submission/data
mkdir -p /home/submission/models

# Download OxfordPets dataset (if not already available)
# We'll use torchvision's built-in dataset loader
# The dataset will be downloaded automatically during training

# Run the training script
python3 train.py \
  --dataset oxford_pets \
  --model resnet18 \
  --image_size 224 \
  --patch_size 8 \
  --layers 5 \
  --channels 3 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --epochs 100 \
  --output_dir /home/submission/results

# Evaluate the model
python3 evaluate.py \
  --dataset oxford_pets \
  --model resnet18 \
  --checkpoint /home/submission/results/best_model.pth \
  --output_file /home/submission/results/evaluation_results.txt

# Generate final results
python3 generate_results.py \
  --input_file /home/submission/results/evaluation_results.txt \
  --output_file /home/submission/results/final_results.txt

# Print completion message
echo "Reproduction completed. Results saved in /home/submission/results/"