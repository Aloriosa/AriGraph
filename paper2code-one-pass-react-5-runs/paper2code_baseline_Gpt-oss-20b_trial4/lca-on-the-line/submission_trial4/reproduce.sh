#!/usr/bin/env bash
set -euo pipefail

# -------------------------------------------------------------
# 1. Install Python and pip (assumes Python 3.8+ is already available)
# -------------------------------------------------------------
echo "Installing Python dependencies ..."
pip install --upgrade pip
pip install -r requirements.txt

# -------------------------------------------------------------
# 2. Download NLTK WordNet data (required for LCA computations)
# -------------------------------------------------------------
echo "Downloading NLTK WordNet data ..."
python - <<'PYCODE'
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
PYCODE

# -------------------------------------------------------------
# 3. Run the evaluation script
# -------------------------------------------------------------
echo "Running evaluation ..."
python evaluate.py

echo "Done. Results are stored in results.json"