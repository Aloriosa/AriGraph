#!/usr/bin/env bash
set -e

# 1. Install dependencies
pip install -q -U pip
pip install -q -r requirements.txt

# 2. Download pre‑trained DDPM model (CIFAR‑10)
python -c "
import torch
from diffusers import DDPMPipeline
pipe = DDPMPipeline.from_pretrained('google/ddpm-cifar10-32')
pipe.save_pretrained('pretrained_ddpm')
"

# 3. Run training & sampling
python src/ant_trainer.py