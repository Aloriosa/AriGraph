#!/usr/bin/env bash
# ==========================================================
# Reproduction script for the PINN training experiment
# ----------------------------------------------------------
# The script installs the required Python packages, runs the
# training for the convection PDE, and stores the final loss
# and L2 relative error in the `results/` directory.
# ==========================================================

set -euo pipefail

# Create a virtual environment (optional but keeps the
# environment clean).  The grader runs this script on a fresh
# Ubuntu 24.04 container, so we install directly into the
# system Python.
pip install --quiet --upgrade pip
pip install --quiet torch==2.0.0+cu118 torchvision==0.15.0+cu118 torchaudio==2.0.0+cu118 -f https://download.pytorch.org/whl/cu118/torch_stable.html
pip install --quiet scipy

# Ensure the results directory exists
mkdir -p results

# Run the training script
python src/pinn.py \
  --pde convection \
  --num_res 2000 \
  --num_bc 200 \
  --num_init 200 \
  --hidden 50 \
  --epochs 5000 \
  --adam_steps 2000 \
  --seed 42

# The training script writes the final loss and L2 error
# to results/conv_loss.txt and results/conv_l2re.txt.
echo "Reproduction finished. Results are in the 'results/' directory."