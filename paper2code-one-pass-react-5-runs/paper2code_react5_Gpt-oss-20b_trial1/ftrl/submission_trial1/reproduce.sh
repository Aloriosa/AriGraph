#!/usr/bin/env bash
set -euo pipefail

# Ensure deterministic behaviour
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
export CUDA_VISIBLE_DEVICES=0

# Install dependencies
python3 -m pip install --quiet -r requirements.txt

# Pre‑train on Phase 2
python3 src/train.py --mode pretrain --seed 0

# Fine‑tune with vanilla (no auxiliary loss)
python3 src/train.py --mode finetune --method vanilla --seed 0

# Fine‑tune with behavioral cloning (BC)
python3 src/train.py --mode finetune --method bc --seed 0

# Fine‑tune with elastic weight consolidation (EWC)
python3 src.train.py --mode finetune --method ewc --seed 0

# Fine‑tune with kick‑starting (KS)
python3 src/train.py --mode finetune --method ks --seed 0

# Fine‑tune with episodic memory (EM)
python3 src/train.py --mode finetune --method em --seed 0

echo "All experiments completed. Results are in results_<method>.json files."