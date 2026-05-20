#!/usr/bin/env bash
set -euo pipefail

# 1. Install system dependencies
apt-get update -y
apt-get install -y python3 python3-pip

# 2. Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Verify CUDA availability
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# 4. Run experiments for a range of lengths and memory vector counts
#    (the script will automatically create CSV files in `results/`).
python3 compress.py \
    --lengths 16 32 64 128 256 \
    --k 1 2 4 8 \
    --epochs 300 \
    --verbose \
    --output-dir results

echo "All experiments completed. Results are in the 'results' folder."