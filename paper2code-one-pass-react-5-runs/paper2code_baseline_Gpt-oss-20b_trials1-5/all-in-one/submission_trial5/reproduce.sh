#!/usr/bin/env bash
#
#  Reproduction script for the Simformer paper
#
#  This script:
#   1. Installs the required Python packages
#   2. Clones the official Simformer repository
#   3. Installs Simformer locally
#   4. Trains a Simformer on the Gaussian linear benchmark task
#   5. Runs inference on a few synthetic observations
#   6. Stores results in ./results/
#
#  The script is designed to run in a clean Ubuntu 24.04 container
#  with the NVIDIA container toolkit available.
#
#  Author: OpenAI ChatGPT
#  Date: 2026-03-14

set -euo pipefail

# Helper: print a message
log() {
  echo -e "\e[1;32m[INFO] $1\e[0m"
}

# ---------------------------------------------
# 1. Install Python 3.10 (if not already installed)
# ---------------------------------------------
log "Installing Python 3.10..."
apt-get update -qq
apt-get install -y --no-install-recommends python3.10 python3.10-venv python3.10-dev

# ---------------------------------------------
# 2. Create a virtual environment
# ---------------------------------------------
VENV=venv
log "Creating virtual environment $VENV..."
python3.10 -m venv $VENV
source $VENV/bin/activate

# ---------------------------------------------
# 3. Upgrade pip
# ---------------------------------------------
log "Upgrading pip..."
pip install --upgrade pip

# ---------------------------------------------
# 4. Install packages from requirements.txt if present
# ---------------------------------------------
if [[ -f requirements.txt ]]; then
  log "Installing packages from requirements.txt ..."
  pip install -r requirements.txt
else
  log "No requirements.txt found, installing packages directly ..."
  pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 torchaudio==2.1.2+cu121 \
    hydra-core==1.3.2 sbi==0.15.0 tqdm==4.66.5 einops==0.7.0 jax==0.4.25 jaxlib==0.4.25 \
    transformers==4.41.2 omegaconf==2.3.0 scikit-learn==1.5.0 numpy==1.26.4 pandas==2.2.2
fi

# ---------------------------------------------
# 5. Clone the official Simformer repository
# ---------------------------------------------
SIMFORER_REPO=simformer
if [[ -d $SIMFORER_REPO ]]; then
  log "Simformer repo already exists, pulling latest changes ..."
  cd $SIMFORER_REPO
  git pull
  cd ..
else
  log "Cloning Simformer repository ..."
  git clone https://github.com/mackelab/simformer.git
fi

# ---------------------------------------------
# 6. Install Simformer locally
# ---------------------------------------------
log "Installing Simformer locally ..."
cd $SIMFORER_REPO
pip install -e .
cd ..

# ---------------------------------------------
# 7. Create a simple training config for Gaussian linear task
# ---------------------------------------------
CONFIG_DIR=simformer/configs
TRAIN_CONFIG=$CONFIG_DIR/gaussian_linear.yaml
if [[ ! -f $TRAIN_CONFIG ]]; then
  log "Creating a minimal training config $TRAIN_CONFIG ..."
  cat <<EOF > $TRAIN_CONFIG
# Minimal training config for Gaussian Linear benchmark
trainer:
  epochs: 10
  batch_size: 256
  learning_rate: 1e-3
  log_interval: 100
  seed: 42
  num_simulations: 10000   # number of simulator draws
model:
  name: Simformer
  transformer:
    num_layers: 6
    num_heads: 4
    d_model: 128
    d_ff: 256
    dropout: 0.1
    attention_mask: "dense"
  diffusion:
    type: VESDE
    sigma_min: 0.0001
    sigma_max: 15.0
    beta_min: 0.01
    beta_max: 10.0
  device: "cuda"  # will fall back to cpu if no GPU
data:
  task: gaussian_linear
  num_params: 10
  num_data: 10
EOF
fi

# ---------------------------------------------
# 8. Train the Simformer
# ---------------------------------------------
log "Starting training ..."
python -m simformer.train --config $TRAIN_CONFIG

# ---------------------------------------------
# 9. Run inference on a few synthetic observations
# ---------------------------------------------
log "Running inference ..."
# Create a simple inference config
INFER_CONFIG=$CONFIG_DIR/gaussian_linear_inference.yaml
cat <<EOF > $INFER_CONFIG
model:
  checkpoint: "outputs/latest.ckpt"   # will be created by training
  device: "cuda"
data:
  task: gaussian_linear
  num_samples: 1000
  num_inference_samples: 1000
  seed: 123
EOF
python -m simformer.inference --config $INFER_CONFIG

# ---------------------------------------------
# 10. Copy results to a public directory
# ---------------------------------------------
log "Collecting results ..."
mkdir -p results
cp -r outputs/* results/

log "Reproduction finished. Results are in ./results/."

# ---------------------------------------------
# 11. Deactivate virtual environment
# ---------------------------------------------
deactivate