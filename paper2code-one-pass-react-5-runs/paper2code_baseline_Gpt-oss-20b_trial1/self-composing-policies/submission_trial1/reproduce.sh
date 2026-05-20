#!/usr/bin/env bash
# ------------------------------------------------------------
# Reproduction script for the minimal CompoNet implementation.
# ------------------------------------------------------------

set -euo pipefail

# Install Python 3.8+ (the container already has it)
# Create a virtual environment to avoid polluting the system
VENV_DIR="${HOME}/.venv_componet"
if [ ! -d "${VENV_DIR}" ]; then
    python3 -m venv "${VENV_DIR}"
fi
source "${VENV_DIR}/bin/activate"

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Run the training script
python -u componet/train.py

echo "Reproduction finished. Check results.txt for performance metrics."