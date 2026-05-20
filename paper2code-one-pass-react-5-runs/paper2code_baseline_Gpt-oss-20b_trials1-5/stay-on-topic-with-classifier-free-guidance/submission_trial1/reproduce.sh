#!/usr/bin/env bash
set -euo pipefail

# -------------------------------------------------------------
# 1. Install dependencies
# -------------------------------------------------------------
echo "Installing dependencies..."
# We prefer the CPU version of torch for portability, but it will automatically
# use the GPU if available.  The container already has CUDA installed.
pip install --quiet torch==2.3.0+cu118 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install --quiet transformers==4.41.0

# -------------------------------------------------------------
# 2. Create output directory
# -------------------------------------------------------------
OUTPUT_DIR="outputs"
mkdir -p "$OUTPUT_DIR"

# -------------------------------------------------------------
# 3. Run the inference script
# -------------------------------------------------------------
echo "Running CFG inference..."
python cfg_inference.py \
  --model_name distilgpt2 \
  --prompts_file prompts.txt \
  --output_dir "$OUTPUT_DIR" \
  --gamma 1.0 1.5

# -------------------------------------------------------------
# 4. Summary
# -------------------------------------------------------------
echo "Reproduction finished. Generated files are in: $OUTPUT_DIR"
echo "Contents:"
ls -l "$OUTPUT_DIR"