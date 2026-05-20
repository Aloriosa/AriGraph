#!/usr/bin/env bash
set -euo pipefail

# 1️⃣ Install dependencies
python -m pip install -q --upgrade pip
python -m pip install -q torch==2.0.1 numpy scikit-learn

# 2️⃣ Run the training & sampling pipeline
python - <<'PY'
import torch
from simformer.train import train_and_sample

train_and_sample(
    num_simulations=100_000,
    epochs=10,
    batch_size=1024,
    num_steps=500,
    output_path="samples.csv",
)
PY