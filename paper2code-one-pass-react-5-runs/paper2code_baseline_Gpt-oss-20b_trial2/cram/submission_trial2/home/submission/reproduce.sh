#!/usr/bin/env bash
set -euo pipefail

# 1. Install system packages
apt-get update
apt-get install -y python3 python3-pip git

# 2. Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Run the training and evaluation pipeline
mkdir -p outputs
python scripts/train_mem_vectors.py \
    --model_name gpt2 \
    --text_file data/sample_texts.txt \
    --output_dir outputs \
    --max_steps 500

echo "Reproduction finished. Results stored in outputs/results.json"