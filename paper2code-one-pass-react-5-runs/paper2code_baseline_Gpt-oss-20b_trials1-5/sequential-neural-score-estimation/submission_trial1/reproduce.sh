#!/usr/bin/env bash
set -euo pipefail

# Install dependencies
echo "Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet torch==2.1.0+cu118 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu118
pip install --quiet numpy scipy scikit-learn tqdm matplotlib

# Run the main script
echo "Running the reproduction script..."
python main.py --dataset two_moons \
               --rounds 5 \
               --samples-per-round 2000 \
               --sigma-min 0.01 \
               --sigma-max 1.0 \
               --epsilon 0.05 \
               --batch-size 256 \
               --max-epochs 30 \
               --patience 5
echo "Reproduction finished. Posterior samples saved to 'posterior_samples.npy'."