#!/usr/bin/env bash
set -euo pipefail

# Install Python packages
echo "Installing Python packages..."
pip install -q -U -r requirements.txt

# Generate random‑word text files
echo "Generating random‑word text files..."
python generate_random_texts.py \
    --vocab_file random_word_list.txt \
    --output_dir random_texts \
    --lengths 64 128 256 512 \
    --num_files 2

# Run compression experiments on the sample texts
echo "Running compression on sample texts..."
python compress_multiple.py \
    --text_dir sample_texts \
    --model_name distilgpt2 \
    --mem_counts 1 2 4 8 \
    --steps 2000 \
    --lr 1e-2 \
    --threshold 0.99

# Run compression experiments on the random‑word texts
echo "Running compression on random‑word texts..."
python compress_multiple.py \
    --text_dir random_texts \
    --model_name distilgpt2 \
    --mem_counts 1 2 4 8 \
    --steps 2000 \
    --lr 1e-2 \
    --threshold 0.99

echo "All experiments completed. Results are in the 'output/' directory."