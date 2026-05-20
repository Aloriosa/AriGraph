#!/usr/bin/env bash
set -e

# Update the system and install Python3
apt-get update -qq
apt-get install -y python3-pip

# Install Python dependencies
pip3 install --upgrade pip
pip3 install -r requirements.txt

# Download the WordNet corpus for nltk
python3 - <<'PY'
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
PY

# Run the evaluation script
python3 eval.py