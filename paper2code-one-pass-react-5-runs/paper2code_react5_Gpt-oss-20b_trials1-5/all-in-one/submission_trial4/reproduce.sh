#!/usr/bin/env bash
set -euo pipefail

# Install required Python packages
pip install --upgrade pip
pip install -q -U -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
    jax[cuda]==0.4.22 \
    jaxlib==0.4.22+cuda121 \
    flax==0.8.1 \
    optax==0.1.9 \
    numpy==1.26.4 \
    tqdm==4.66.2

# Train Simformer
python -m src.train_simformer

# Evaluate: sample posterior and report MSE
python -m src.evaluate