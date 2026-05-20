#!/usr/bin/env bash
set -euo pipefail
echo "==> Installing system dependencies"
apt-get update -y
apt-get install -y python3 python3-pip

echo "==> Installing Python packages"
pip install --upgrade pip
pip install -r requirements.txt

# Run training and evaluation for each dataset
echo "==> Running GSM8K"
python train.py --dataset gsm8k --adapter_size distilbert-base-uncased --epochs 3 --k 5 --batch_size 8

echo "==> Running StrategyQA"
python train.py --dataset strategyqa --adapter_size distilbert-base-uncased --epochs 3 --k 5 --batch_size 8

echo "==> Running TruthfulQA"
python train.py --dataset truthfulqa --adapter_size distilbert-base-uncased --epochs 3 --k 5 --batch_size 8

echo "==> Running ScienceQA"
python train.py --dataset scienceqa --adapter_size distilbert-base-uncased --epochs 3 --k 5 --batch_size 8