#!/bin/bash

# Set up environment
apt-get update && apt-get install -y python3 python3-pip git libgl1-mesa-glx libglib2.0-0

# Install required packages
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install transformers datasets scikit-learn numpy matplotlib tqdm pandas wandb

# Create directory structure
mkdir -p /home/submission/data
mkdir -p /home/submission/models
mkdir -p /home/submission/results

# Download and prepare datasets
python3 prepare_datasets.py

# Train SEMA model on Split CIFAR-100
python3 train_sema.py \
    --dataset split_cifar100 \
    --model vit_b_16 \
    --num_tasks 10 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --adapter_dim 64 \
    --expansion_threshold 0.1 \
    --max_adapters_per_layer 3 \
    --epochs 10 \
    --output_dir /home/submission/models/sema_cifar100

# Evaluate SEMA model
python3 evaluate_sema.py \
    --model_path /home/submission/models/sema_cifar100 \
    --dataset split_cifar100 \
    --num_tasks 10 \
    --output_file /home/submission/results/sema_cifar100_results.csv

# Train SEMA model on Split Tiny ImageNet
python3 train_sema.py \
    --dataset split_tiny_imagenet \
    --model vit_b_16 \
    --num_tasks 20 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --adapter_dim 64 \
    --expansion_threshold 0.1 \
    --max_adapters_per_layer 3 \
    --epochs 10 \
    --output_dir /home/submission/models/sema_tiny_imagenet

# Evaluate SEMA model on Split Tiny ImageNet
python3 evaluate_sema.py \
    --model_path /home/submission/models/sema_tiny_imagenet \
    --dataset split_tiny_imagenet \
    --num_tasks 20 \
    --output_file /home/submission/results/sema_tiny_imagenet_results.csv

# Generate comparison results with baselines
python3 compare_baselines.py \
    --sema_cifar100_results /home/submission/results/sema_cifar100_results.csv \
    --sema_tiny_imagenet_results /home/submission/results/sema_tiny_imagenet_results.csv \
    --output_file /home/submission/results/baseline_comparison.csv

# Generate visualization of adapter expansion
python3 visualize_expansion.py \
    --model_path /home/submission/models/sema_cifar100 \
    --output_file /home/submission/results/expansion_visualization.png

# Report results
echo "Reproduction complete!"
echo "SEMA results saved to /home/submission/results/"
echo "Model checkpoints saved to /home/submission/models/"
echo "Visualizations saved to /home/submission/results/"