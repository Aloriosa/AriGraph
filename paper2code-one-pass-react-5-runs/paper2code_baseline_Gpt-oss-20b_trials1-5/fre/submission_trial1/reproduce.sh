#!/usr/bin/env bash
set -e

# 1. Install Python dependencies
python3 -m pip install --quiet -r requirements.txt

# 2. (Optional) Install torch with GPU support if available
#    Uncomment the following line if you want CUDA support:
#    python3 -m pip install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 3. Run the FRE training script
python3 fre/train_fre.py

# 4. Show the results
echo "=== FRE Results ==="
cat results/fre_results.csv