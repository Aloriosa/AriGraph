#!/usr/bin/env bash
# reproduce.sh – reproducible training & evaluation of FARE on CLIP

set -euo pipefail

echo "=== Updating apt and installing Python3 ==="
sudo apt-get update -y
sudo apt-get install -y python3 python3-pip

echo "=== Installing Python dependencies ==="
pip install -r requirements.txt

echo "=== Training the FARE‑CLIP model ==="
python train_fare.py

echo "=== Evaluating the trained model ==="
python eval_fare.py