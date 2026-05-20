#!/usr/bin/env bash
set -e

# 1. Install Python dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# 2. Create required directories
mkdir -p data/target
mkdir -p output/generated_images

# 3. Download 10 target images (public domain, unsplash)
echo "Downloading 10 target images..."
python download_target_images.py

# 4. Train the model
echo "Starting training..."
python -m src.train --config config.yaml

# 5. Generate images
echo "Generating images..."
python -m src.generate --config config.yaml

echo "Reproduction finished. Results are in output/generated_images/"