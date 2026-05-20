#!/usr/bin/env bash
# reproducibility script for the compression demo
set -e

# Install Python packages
apt-get update && apt-get install -y python3-pip
pip install --quiet --upgrade pip

# Install PyTorch with CUDA 12.1 (container should have CUDA 12.1)
pip install --quiet torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0+cu121 \
    -f https://download.pytorch.org/whl/torch_stable.html
pip install --quiet transformers==4.40.0

# Create a small sample text file
cat > sample.txt <<'EOF'
The quick brown fox jumps over the lazy dog.
EOF

# Run the compression script
python3 compress.py --model_name gpt2 --text_file sample.txt --num_mem 1 \
    --max_steps 2000 --lr 1e-2 --output_dir output --threshold 0.99

echo "Reproduction finished."