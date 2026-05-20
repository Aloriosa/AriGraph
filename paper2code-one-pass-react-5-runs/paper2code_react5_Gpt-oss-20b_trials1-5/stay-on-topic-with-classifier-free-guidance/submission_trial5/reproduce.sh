#!/usr/bin/env bash
# reproducibility script for *Stay on topic with Classifier‑Free Guidance*
#
# The script installs the required Python dependencies, downloads the GPT‑2
# medium checkpoint, and runs the example scripts that demonstrate:
#   1. a small LAMBADA benchmark (vanilla vs. CFG)
#   2. a chain‑of‑thought demo
#   3. an assistant‑style demo with negative prompting
#
# All output is written to the console.  The script is intentionally
# lightweight and can run on a fresh Ubuntu 24.04 container with an
# NVIDIA A10 GPU.

set -euo pipefail

# 1. Install Python 3 and pip
apt-get update && apt-get install -y python3-pip git

# 2. Install Python packages
pip install --upgrade pip
pip install --no-cache-dir torch==2.3.0+cu118 torchvision==0.18.0+cu118 torchaudio==2.3.0+cu118 \
    transformers==4.40.0 datasets==2.20.0 tqdm==4.66.5

# 3. Run the demonstration scripts
echo "=== Running benchmark on LAMBADA ==="
python benchmark_lambada.py

echo "=== Running chain‑of‑thought demo ==="
python chain_of_thought_demo.py

echo "=== Running assistant demo with negative prompting ==="
python assistant_demo.py

echo "=== Reproduction finished ==="