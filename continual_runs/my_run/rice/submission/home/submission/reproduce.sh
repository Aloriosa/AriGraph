#!/bin/bash

# Set up environment
apt-get update && apt-get install -y python3 python3-pip git libgl1-mesa-glx libglib2.0-0

# Install required packages
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install gymnasium[all] numpy matplotlib scikit-learn wandb tqdm pyyaml

# Create directory structure
mkdir -p /home/submission/data
mkdir -p /home/submission/models
mkdir -p /home/submission/results

# Download or generate synthetic offline trajectories data
python3 generate_trajectories.py

# Train the pre-trained PPO policy (guide policy)
python3 train_ppo.py \
    --env_name HalfCheetah-v4 \
    --num_steps 300000 \
    --seed 42 \
    --model_path models/pretrained_ppo.pth

# Extract critical states using StateMask explanation method
python3 extract_critical_states.py \
    --trajectory_path data/offline_trajectories.pkl \
    --pretrained_policy_path models/pretrained_ppo.pth \
    --critical_states_path data/critical_states.pkl \
    --top_k 10 \
    --explanation_method integrated_gradients

# Refine policy using RICE algorithm
python3 refine_policy_rice.py \
    --env_name HalfCheetah-v4 \
    --pretrained_policy_path models/pretrained_ppo.pth \
    --critical_states_path data/critical_states.pkl \
    --mixing_ratio 0.5 \
    --num_steps 100000 \
    --seed 42 \
    --output_path results/rice_refined_policy.pth \
    --results_file results/rice_results.csv

# Evaluate refined policy against baselines
python3 evaluate_baselines.py \
    --env_name HalfCheetah-v4 \
    --pretrained_policy_path models/pretrained_ppo.pth \
    --rice_policy_path results/rice_refined_policy.pth \
    --num_episodes 100 \
    --num_seeds 5 \
    --output results/baseline_comparison.csv

# Generate visualization of critical states
python3 visualize_critical_states.py \
    --critical_states_path data/critical_states.pkl \
    --output results/critical_states_visualization.png

# Report results
echo "Reproduction complete!"
echo "Final policy saved to results/rice_refined_policy.pth"
echo "Results saved to results/rice_results.csv and results/baseline_comparison.csv"
echo "Critical states visualization saved to results/critical_states_visualization.png"