#!/usr/bin/env bash
set -euo pipefail

echo "Installing dependencies..."
pip install -q -r requirements.txt

# ---- Baseline (independent coupling) ----
echo "=== Training baseline model (in‑painting) ==="
python train.py --task baseline --epochs 5 --batch_size 128 --lr 2e-4 --sigma 0.1 --save_dir baseline

# ---- Coupled model (data‑dependent coupling) ----
echo "=== Training coupled model (in‑painting) ==="
python train.py --task inpaint --epochs 5 --batch_size 128 --lr 2e-4 --sigma 0.1 --save_dir coupled

# ---- Sampling ----
echo "=== Generating samples from baseline ==="
python train.py --sample --ckpt baseline/model_best.pt --num_samples 500 --sample_batch 32 --out_dir samples_baseline

echo "=== Generating samples from coupled ==="
python train.py --sample --ckpt coupled/model_best.pt --num_samples 500 --sample_batch 32 --out_dir samples_coupled

# ---- Evaluation (FID) ----
echo "=== Computing FID for baseline ==="
python eval.py --gen_dir samples_baseline --output baseline_fid.txt --num_samples 500

echo "=== Computing FID for coupled ==="
python eval.py --gen_dir samples_coupled --output coupled_fid.txt --num_samples 500

echo "Reproduction finished. FID scores:"
cat baseline_fid.txt
cat coupled_fid.txt