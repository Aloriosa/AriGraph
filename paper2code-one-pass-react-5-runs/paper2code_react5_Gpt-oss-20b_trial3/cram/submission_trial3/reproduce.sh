#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------ #
# 1. Install system packages and Python
# ------------------------------------------------------------ #
apt-get update && apt-get install -y python3 python3-pip

# ------------------------------------------------------------ #
# 2. Install PyTorch (CPU+CUDA)
# ------------------------------------------------------------ #
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121

# ------------------------------------------------------------ #
# 3. Install remaining dependencies
# ------------------------------------------------------------ #
pip install -r requirements.txt

# ------------------------------------------------------------ #
# 4. Run the compression experiments
# ------------------------------------------------------------ #
python compress.py --config config.json

echo "Reproduction finished. Results are in results.json"