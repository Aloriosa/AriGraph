#!/usr/bin/env bash
set -euo pipefail

# 1. Install dependencies
python3 -m pip install --quiet --upgrade pip
python3 -m pip install --quiet -r requirements.txt

# 2. Train TSNPSE model
python3 src/train_npse.py \
    --batch-size 256 \
    --batches 10000 \
    --epochs-per-round 3 \
    --rounds 5 \
    --out-dir output

# 3. Sample from posterior using trained model
python3 src/sample.py \
    --num-samples 5000 \
    --model output/score_net.pt \
    --out-dir output

# 4. Evaluate results
python3 evaluate.py --out-dir output

# 5. Inform user
echo "Training, sampling, and evaluation completed."
echo "Posterior samples stored in: output/posterior_samples.npy"
echo "Metrics stored in: output/metrics.txt"