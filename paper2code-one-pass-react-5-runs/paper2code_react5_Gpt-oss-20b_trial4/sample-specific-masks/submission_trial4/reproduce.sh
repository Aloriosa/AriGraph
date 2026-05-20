#!/usr/bin/env bash
set -euo pipefail

echo "===== Step 1: Install dependencies ====="
apt-get update -y
apt-get install -y python3 python3-pip
pip3 install -r requirements.txt

echo "===== Step 2: Run training ====="
python3 -m src.main

echo "===== Done! ====="