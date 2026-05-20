#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# 1. Install dependencies
# ------------------------------------------------------------
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# ------------------------------------------------------------
# 2. Create output directories
# ------------------------------------------------------------
mkdir -p outputs checkpoints logs

# ------------------------------------------------------------
# 3. Run training and evaluation
# ------------------------------------------------------------
echo "Running training and evaluation..."
python -m src.trainer \
    --dataset strategyqa \
    --train_size 200 \
    --max_seq_len 512 \
    --batch_size 8 \
    --epochs 5 \
    --learning_rate 5e-6 \
    --adapter_size 0.1 \
    --output_dir checkpoints/adapter_strategyqa \
    --log_dir logs

echo "Running inference on test set..."
python -m src.inference \
    --dataset strategyqa \
    --adapter_path checkpoints/adapter_strategyqa/adapter.pt \
    --llm_name mistralai/Mixtral-8x7B-v0.1 \
    --beam_size 3 \
    --candidate_num 5 \
    --max_seq_len 512 \
    --log_dir logs

echo "Done. Results are in logs/eval.log"