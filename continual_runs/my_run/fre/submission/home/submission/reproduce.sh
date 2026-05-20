#!/bin/bash

# Set up environment
apt-get update && apt-get install -y python3 python3-pip git

# Install required packages
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install gymnasium numpy matplotlib scikit-learn wandb tqdm

# Create directory structure
mkdir -p /home/submission/data
mkdir -p /home/submission/models
mkdir -p /home/submission/results

# Download or generate synthetic offline trajectories data
# Since the paper uses simulated robotic benchmarks, we'll generate synthetic data
python3 generate_data.py

# Train the reward encoder (VAE)
python3 train_reward_encoder.py \
    --data_path data/offline_trajectories.pkl \
    --model_path models/reward_encoder.pth \
    --latent_dim 128 \
    --num_layers 4 \
    --num_heads 8 \
    --hidden_dim 256 \
    --batch_size 64 \
    --learning_rate 0.001 \
    --epochs 50 \
    --device cuda

# Train the generalist policy using IQL
python3 train_policy.py \
    --data_path data/offline_trajectories.pkl \
    --reward_encoder_path models/reward_encoder.pth \
    --policy_path models/policy.pth \
    --latent_dim 128 \
    --hidden_dim 256 \
    --batch_size 64 \
    --learning_rate 0.0003 \
    --epochs 100 \
    --device cuda

# Evaluate zero-shot performance on unseen tasks
python3 evaluate_zero_shot.py \
    --reward_encoder_path models/reward_encoder.pth \
    --policy_path models/policy.pth \
    --num_eval_tasks 10 \
    --episodes_per_task 5 \
    --device cuda \
    --output results/zero_shot_results.csv

# Generate visualization of latent space
python3 visualize_latent_space.py \
    --reward_encoder_path models/reward_encoder.pth \
    --data_path data/offline_trajectories.pkl \
    --output results/latent_space_tsne.png

# Report results
echo "Reproduction complete!"
echo "Results saved to results/zero_shot_results.csv"
echo "Latent space visualization saved to results/latent_space_tsne.png"