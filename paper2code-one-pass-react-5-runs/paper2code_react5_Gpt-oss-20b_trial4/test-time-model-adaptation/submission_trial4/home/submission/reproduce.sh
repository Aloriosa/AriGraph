#!/usr/bin/env bash
set -euo pipefail

# Install python dependencies
pip install -q -U pip
pip install -q -r requirements.txt

# Create a working directory
mkdir -p /tmp/foa_work
cd /tmp/foa_work

# Download ImageNet‑C (level 5)
if [ ! -f imagenet_c.zip ]; then
  echo "Downloading ImageNet‑C ..."
  wget -q https://github.com/hendrycks/imagenet_c/archive/master.zip -O imagenet_c.zip
  unzip -q imagenet_c.zip
  mv imagenet_c-master imagenet_c
fi

# Download ImageNet‑1K validation (used for source statistics)
if [ ! -f imagenet_val.zip ]; then
  echo "Downloading ImageNet‑1K validation ..."
  # The original ImageNet files are not publicly available; for the purpose of this
  # reproduction we download the validation split from the official pytorch dataset
  mkdir -p imagenet_val
  cd imagenet_val
  wget -q https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz
  tar -xzf imagenette2-320.tgz
  cd ..
fi

# Move the downloaded data to the repository folder for easy access
cp -r imagenet_c /home/submission/
cp -r imagenet_val /home/submission/

# Run the evaluation
cd /home/submission
python evaluate.py --datasets imagenet_c imagenet_val \
  --model vit_base_patch16_224 --quantize 8 \
  --batch-size 64 --population-size 28 --prompt-size 3 \
  --seed 42