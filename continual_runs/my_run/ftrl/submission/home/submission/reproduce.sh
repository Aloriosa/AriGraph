#!/bin/bash

# Set up environment
apt-get update && apt-get install -y python3 python3-pip git libgl1-mesa-glx libglib2.0-0

# Install required packages
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install gymnasium[all] numpy matplotlib scikit-learn wandb tqdm pyyaml stable-baselines3[extra] opencv-python

# Create directory structure
mkdir -p /home/submission/data
mkdir -p /home/submission/models
mkdir -p /home/submission/results

# Download or generate synthetic offline trajectories data
# For Montezuma's Revenge, we'll use a pre-trained policy on a simpler task
python3 generate_pretrained_policy.py --env_name MontezumaRevenge-v4 --num_steps 200000 --seed 42 --model_path models/pretrained_montezuma.pth

# Fine-tuning with different knowledge retention techniques
echo "Running vanilla fine-tuning..."
python3 fine_tune.py \
    --env_name MontezumaRevenge-v4 \
    --pretrained_model_path models/pretrained_montezuma.pth \
    --method vanilla \
    --num_steps 100000 \
    --seed 42 \
    --output_path results/vanilla_finetune.pth \
    --results_file results/vanilla_results.csv

echo "Running fine-tuning with behavioral cloning (BC)..."
python3 fine_tune.py \
    --env_name MontezumaRevenge-v4 \
    --pretrained_model_path models/pretrained_montezuma.pth \
    --method bc \
    --num_steps 100000 \
    --seed 42 \
    --output_path results/bc_finetune.pth \
    --results_file results/bc_results.csv

echo "Running fine-tuning with EWC..."
python3 fine_tune.py \
    --env_name MontezumaRevenge-v4 \
    --pretrained_model_path models/pretrained_montezuma.pth \
    --method ewc \
    --num_steps 100000 \
    --seed 42 \
    --output_path results/ewc_finetune.pth \
    --results_file results/ewc_results.csv

echo "Running fine-tuning with kickstarting (KS)..."
python3 fine_tune.py \
    --env_name MontezumaRevenge-v4 \
    --pretrained_model_path models/pretrained_montezuma.pth \
    --method ks \
    --num_steps 100000 \
    --seed 42 \
    --output_path results/ks_finetune.pth \
    --results_file results/ks_results.csv

echo "Running training from scratch for comparison..."
python3 train_from_scratch.py \
    --env_name MontezumaRevenge-v4 \
    --num_steps 100000 \
    --seed 42 \
    --output_path results/scratch.pth \
    --results_file results/scratch_results.csv

# Evaluate all policies
echo "Evaluating all policies..."
python3 evaluate_baselines.py \
    --env_name MontezumaRevenge-v4 \
    --vanilla_path results/vanilla_finetune.pth \
    --bc_path results/bc_finetune.pth \
    --ewc_path results/ewc_finetune.pth \
    --ks_path results/ks_finetune.pth \
    --scratch_path results/scratch.pth \
    --pretrained_path models/pretrained_montezuma.pth \
    --num_episodes 100 \
    --num_seeds 5 \
    --output results/baseline_comparison.csv

# Generate summary report
python3 generate_summary.py \
    --results_dir results \
    --output results/final_summary.txt

# Report results
echo "Reproduction complete!"
echo "Final results saved to results/baseline_comparison.csv and results/final_summary.txt"