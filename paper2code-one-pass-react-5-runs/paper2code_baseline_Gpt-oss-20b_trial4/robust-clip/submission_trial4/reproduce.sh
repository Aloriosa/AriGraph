#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# 1. Install Python dependencies
# ------------------------------------------------------------
pip install -q -r requirements.txt

# ------------------------------------------------------------
# 2. Download the CLIP model (this will cache the weights)
# ------------------------------------------------------------
python -c "import clip, torch; model, _ = clip.load('ViT-B/32', device='cpu')"

# ------------------------------------------------------------
# 3. Fine‑tune with FARE
# ------------------------------------------------------------
python src/clip_finetune.py \
    --dataset cifar10 \
    --epochs 2 \
    --batch-size 128 \
    --lr 1e-4 \
    --wd 1e-4 \
    --adv-steps 10 \
    --eps 4/255

# ------------------------------------------------------------
# 4. Evaluate clean & adversarial accuracy
# ------------------------------------------------------------
python src/evaluate.py \
    --model finetuned_clip.pt \
    --dataset cifar10 \
    --eps 4/255