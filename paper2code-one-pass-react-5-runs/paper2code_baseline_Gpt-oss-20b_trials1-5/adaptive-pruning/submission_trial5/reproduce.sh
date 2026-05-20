#!/usr/bin/env bash
set -euo pipefail

# 1. Install minimal packages
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 2. Run training
python train_apt.py \
    --model_name bert-base-uncased \
    --task sst2 \
    --epochs 3 \
    --batch_size 32 \
    --lr 2e-5 \
    --output_dir ./results \
    --init_rank 8 \
    --max_rank 32 \
    --rank_increase_epochs 1 2 \
    --head_sparsity 0.4

echo "Reproduction finished. Results are in ./results/results.json"