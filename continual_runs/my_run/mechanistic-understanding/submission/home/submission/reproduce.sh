#!/bin/bash

# Set up environment
apt-get update && apt-get install -y python3 python3-pip git

# Install required packages
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install transformers datasets accelerate wandb tqdm evaluate

# Create directory structure
mkdir -p /home/submission/data
mkdir -p /home/submission/models
mkdir -p /home/submission/results

# Download the synthetic pairwise preference dataset (simulating the 200MB dataset from paper)
# In a real implementation, we would use the actual dataset from the paper's repository
python3 generate_pairwise_dataset.py

# Train DPO on GPT-2 Medium
python3 train_dpo.py \
    --model_name gpt2-medium \
    --dataset_path data/pairwise_dataset.pkl \
    --output_dir models/gpt2_medium_dpo \
    --learning_rate 5e-6 \
    --batch_size 16 \
    --num_epochs 1 \
    --beta 0.1 \
    --max_length 512 \
    --device cuda

# Train DPO on Llama2-7B
python3 train_dpo.py \
    --model_name meta-llama/Llama-2-7b-chat-hf \
    --dataset_path data/pairwise_dataset.pkl \
    --output_dir models/llama2_7b_dpo \
    --learning_rate 5e-6 \
    --batch_size 16 \
    --num_epochs 1 \
    --beta 0.1 \
    --max_length 512 \
    --device cuda

# Evaluate toxicity reduction using PerspectiveAPI
python3 evaluate_toxicity.py \
    --model_path models/gpt2_medium_dpo \
    --dataset_path data/pairwise_dataset.pkl \
    --output results/gpt2_medium_toxicity_results.csv \
    --device cuda

python3 evaluate_toxicity.py \
    --model_path models/llama2_7b_dpo \
    --dataset_path data/pairwise_dataset.pkl \
    --output results/llama2_7b_toxicity_results.csv \
    --device cuda

# Generate evaluation report
python3 generate_report.py \
    --gpt2_results results/gpt2_medium_toxicity_results.csv \
    --llama2_results results/llama2_7b_toxicity_results.csv \
    --output results/evaluation_report.md

# Report results
echo "DPO training and evaluation completed!"
echo "GPT-2 Medium toxicity reduction results saved to results/gpt2_medium_toxicity_results.csv"
echo "Llama2-7B toxicity reduction results saved to results/llama2_7b_toxicity_results.csv"
echo "Full evaluation report saved to results/evaluation_report.md"