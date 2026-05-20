#!/usr/bin/env bash
set -euo pipefail

# -------------------------------------------------
# 1. Install dependencies
# -------------------------------------------------
apt-get update && apt-get install -y python3-pip
python3 -m pip install --upgrade pip
python3 -m pip install --quiet transformers torch tqdm

# -------------------------------------------------
# 2. Prepare a tiny example text
# -------------------------------------------------
cat <<'EOF' > sample.txt
Once upon a midnight dreary, while I pondered, weak and weary, I was reading an old piece of literature.
EOF

# -------------------------------------------------
# 3. Train a single memory vector
# -------------------------------------------------
python3 compress_mem.py \
    --model_name gpt2 \
    --text_file sample.txt \
    --k 1 \
    --output_dir output \
    --max_steps 200 \
    --lr 1e-3

# -------------------------------------------------
# 4. Evaluate reconstruction quality
# -------------------------------------------------
python3 evaluate.py \
    --model_name gpt2 \
    --output_dir output