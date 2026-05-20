#!/usr/bin/env bash
set -e

# ---- 1. Install dependencies -----------------------------------------------
pip install -r requirements.txt

# ---- 2. Download NLTK WordNet ------------------------------------------------
python - <<'PY'
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
PY

# ---- 3. Check for ImageNet validation set -----------------------------------
if [ ! -d "data/imagenet/val" ]; then
  echo "❌ ImageNet validation set not found."
  echo "Please download the ImageNet validation set (1.28M images) and place it in data/imagenet/val."
  echo "The folder structure should be:"
  echo "  data/imagenet/val/<synset_id>/<image_files>"
  exit 1
fi

# ---- 4. Run the evaluation --------------------------------------------------
python -m src.main