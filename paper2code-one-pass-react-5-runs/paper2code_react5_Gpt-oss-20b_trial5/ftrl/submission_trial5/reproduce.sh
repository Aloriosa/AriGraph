#!/usr/bin/env bash
set -e

# Install dependencies
pip install --quiet torch==2.0.1 gym==0.26.2 numpy==1.26.4

# Pre‑train policy on phase‑2 (return) only
python train_pretrain.py

# Fine‑tune with Knowledge Retention (BC)
python fine_tune.py --method BC --episodes 500 --lr 1e-3

# Evaluate the fine‑tuned policy
python evaluate.py --model finetune/finetune.pt --episodes 200