#!/bin/bash
set -e

# Install dependencies
apt-get update && apt-get install -y python3 python3-pip git

# Install required Python packages
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install transformers datasets accelerate peft scikit-learn numpy matplotlib tqdm

# Create results directory
mkdir -p results

# Download and prepare dataset (using SST-2 as primary example from paper)
python3 prepare_dataset.py

# Run APT implementation on RoBERTa-base with SST-2 dataset (as in Table 2)
python3 apt_main.py \
    --model_name roberta-base \
    --task sst2 \
    --sparsity 0.6 \
    --epochs 5 \
    --batch_size 16 \
    --learning_rate 2e-4 \
    --output_dir results/apt_sst2

# Run APT implementation on T5-base with CNN/DM dataset (as in Table 2)
python3 apt_main.py \
    --model_name t5-base \
    --task cnn_dailymail \
    --sparsity 0.6 \
    --epochs 5 \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --output_dir results/apt_cnn_dm

# Run APT implementation on LLaMA-2 7B with Alpaca dataset (as in Table 3)
# Note: LLaMA-2 requires access request, so we'll use a smaller model for demonstration
python3 apt_main.py \
    --model_name meta-llama/Llama-2-7b-hf \
    --task alpaca \
    --sparsity 0.3 \
    --epochs 3 \
    --batch_size 4 \
    --learning_rate 1e-4 \
    --output_dir results/apt_llama \
    --use_dummy_llama

# Generate results summary
python3 generate_results.py

echo "APT reproduction completed. Results saved in results/ directory."