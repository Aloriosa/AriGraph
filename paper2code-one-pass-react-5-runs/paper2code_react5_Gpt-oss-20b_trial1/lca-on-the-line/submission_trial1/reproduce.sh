#!/usr/bin/env bash
set -euo pipefail

# Install Python dependencies
python3 -m pip install --upgrade pip
pip install -r requirements.txt

# Ensure NLTK WordNet data is downloaded
python3 -c "import nltk; nltk.download('wordnet')"

# Run the main experiment
python3 lca_experiment.py