#!/usr/bin/env bash
set -euo pipefail

# install dependencies – the judge already has NVIDIA drivers, but we only need CPU
python -m pip install --quiet torch==2.0 torchvision==0.15 tqdm

# run the algorithm
python src/lbcs.py \
    --dataset mnist \
    --k 2000 \
    --epsilon 0.1 \
    --T 200 \
    --epochs 5 \
    --final_epochs 10