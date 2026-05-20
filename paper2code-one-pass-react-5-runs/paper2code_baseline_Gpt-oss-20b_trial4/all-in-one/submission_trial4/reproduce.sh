#!/usr/bin/env bash
set -euo pipefail

# -------------------------------------------------------------
# This script installs the required dependencies, clones the
# original Simformer repository, trains the model on a simple
# benchmark task (Gaussian Linear) and writes the results to
# `results.txt`.  The script is intentionally lightweight
# and can be executed in a fresh Ubuntu 24.04 Docker container
# with an NVIDIA A10 GPU.
# -------------------------------------------------------------

# 1. Install system packages
apt-get update -qq
apt-get install -y -qq python3 python3-pip git

# 2. Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install Python dependencies
pip install --upgrade pip
pip install --no-binary :all: torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install jax jaxlib==0.4.8  # JAX for Simformer
pip install sbi tqdm hydra-core

# 4. Clone the Simformer repository (original implementation)
git clone --depth 1 https://github.com/mackelab/simformer.git
cd simformer

# 5. Checkout a stable commit (the one used in the paper)
git checkout 6f8c2a4  # replace with the commit hash of the paper version

# 6. Install the Simformer package
pip install -e .

# 7. Run a training run on the Gaussian Linear task
#    (the Simformer repo ships with a simple CLI)
python -m simformer.train_gaussian_linear \
    --trainer.batch_size 1000 \
    --trainer.num_epochs 5 \
    --trainer.learning_rate 1e-3 \
    --trainer.save_dir ../results \
    --trainer.eval_interval 1 \
    --trainer.save_every_epoch True \
    > ../train_log.txt 2>&1

# 8. After training, perform inference on a held‑out synthetic dataset
python -m simformer.eval_gaussian_linear \
    --model_dir ../results \
    --num_samples 10000 \
    > ../eval_log.txt 2>&1

# 9. Compute a simple classifier two‑sample test (C2ST) score
python -m simformer.eval_c2st \
    --model_dir ../results \
    --num_samples 5000 \
    --output_path ../results/c2st.txt

# 10. Print a short summary
echo "Training finished. Results are in ../results/"
echo "C2ST score (lower is better, 0.5 is perfect):"
cat ../results/c2st.txt