#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------------
# 1. Install system dependencies
# ------------------------------------------------------------------
apt-get update && apt-get install -y python3 python3-pip git wget unzip

# ------------------------------------------------------------------
# 2. Install Python dependencies
# ------------------------------------------------------------------
pip install --upgrade pip
pip install -r requirements.txt

# ------------------------------------------------------------------
# 3. Download NLTK data (WordNet)
# ------------------------------------------------------------------
python - <<'PY'
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
PY

# ------------------------------------------------------------------
# 4. Download ImageNet validation subset
# ------------------------------------------------------------------
if [ -z "${IMAGENET_DIR:-}" ]; then
    echo "Downloading 1k‑image ImageNet validation subset..."
    mkdir -p data/imagenet
    wget -q https://github.com/pytorch/hub/raw/master/imagenet_val.tar -O data/imagenet/imagenet_val.tar
    tar -xf data/imagenet/imagenet_val.tar -C data/imagenet
    export IMAGENET_DIR=$(realpath data/imagenet)
else
    echo "Using user provided ImageNet directory: ${IMAGENET_DIR}"
fi

# ------------------------------------------------------------------
# 5. Download OOD subsets (tiny versions)
# ------------------------------------------------------------------
python dataset_downloads.py

# ------------------------------------------------------------------
# 6. Run the evaluation
# ------------------------------------------------------------------
# The extracted directories contain a `val` subdirectory; we pass the
# parent directory so that `load_imagenet_split` can find `val/`.
python compute.py \
    --imagenet-dir "$IMAGENET_DIR" \
    --ood-dirs data/imagenet_sketch/imagenet_sketch_val \
               data/imagenet_render/imagenet_render_val \
               data/imagenet_adv/imagenet_adv_val \
               data/objectnet/objectnet_val \
    --batch-size 64 \
    --device cuda \
    --correlate

# ------------------------------------------------------------------
# 7. Done
# ------------------------------------------------------------------
echo "Reproduction finished. Results in results.csv and correlation.csv."