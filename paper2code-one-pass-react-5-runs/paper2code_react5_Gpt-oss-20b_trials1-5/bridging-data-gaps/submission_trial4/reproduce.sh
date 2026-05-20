#!/usr/bin/env bash
set -e

# Install system dependencies
apt-get update
apt-get install -y git-lfs

# Install Python packages
pip install --upgrade pip
pip install -r requirements.txt

# Create output directories
mkdir -p outputs/checkpoints
mkdir -p outputs/generated_images
mkdir -p outputs/evaluation

# Run training
python src/train.py \
    --source cifar10 \
    --target cifar10_subset \
    --seed 42 \
    --epochs 3 \
    --batch-size 64 \
    --learning-rate 5e-5 \
    --adaptor-lr 1e-4 \
    --adaptor-steps 0 \
    --adversarial-steps 10 \
    --adversarial-lr 0.02 \
    --gamma 5 \
    --output-dir outputs

# Generate samples after training
python src/generate.py --checkpoint outputs/checkpoints/fine_tuned.pth \
    --num-samples 1000 \
    --output-dir outputs/generated_images

# Evaluate
python -m pytorch_fid.outputs/generated_images \
    --batch-size 50 \
    --gpu 0 \
    outputs/generated_images \
    --real-labels outputs/target_images.txt > outputs/evaluation/fid.txt

# Compute LPIPS
python -c "
import torch, lpips, glob
from PIL import Image
import numpy as np
def load_images(path):
    imgs = []
    for f in sorted(glob.glob(path+'/*.png')):
        imgs.append(np.array(Image.open(f).convert('RGB'))/255.0)
    return torch.tensor(imgs).permute(0,3,1,2).float()
gen = load_images('outputs/generated_images')
real = load_images('outputs/target_images')
loss = lpips.LPIPS(net='vgg')
print('LPIPS:', loss(gen, real).mean().item())
" > outputs/evaluation/lpips.txt

echo "Reproduction finished. Metrics stored in outputs/evaluation/"