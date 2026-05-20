#!/usr/bin/env bash
set -euo pipefail

# 1. Install system packages
apt-get update && apt-get install -y python3 python3-pip git

# 2. Install Python dependencies
pip install --upgrade pip
pip install torch==2.2.0+cu121 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.40.0 datasets==2.19.1 accelerate==0.26.0 tqdm==4.66.2

# 3. Run the evaluation scripts
echo "=== LAMBADA evaluation (CFG γ=1.5) ==="
python eval_lambada.py --gamma 1.5 > lambada_results.txt
echo "=== GSM‑8K CoT evaluation (CFG γ=1.5) ==="
python eval_gsm8k.py --gamma 1.5 > gsm8k_results.txt
echo "=== Assistant prompt demo (CFG γ=1.5) ==="
python eval_assistant.py > assistant_results.txt

echo "Reproduction finished. Results are in lambada_results.txt, gsm8k_results.txt, assistant_results.txt"