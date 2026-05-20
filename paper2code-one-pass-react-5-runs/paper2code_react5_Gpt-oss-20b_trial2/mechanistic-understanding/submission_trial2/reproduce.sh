#!/usr/bin/env bash
set -euo pipefail

# -------------------------------------------------------------
# 1. Install dependencies
# -------------------------------------------------------------
echo "Installing Python dependencies..."
pip install --quiet -r requirements.txt

# -------------------------------------------------------------
# 2. Train toxic‑comment probe
# -------------------------------------------------------------
echo "Training toxic‑comment probe..."
python -m src.probe_toxicity

# -------------------------------------------------------------
# 3. Extract toxic vectors
# -------------------------------------------------------------
echo "Extracting toxic vectors..."
python -m src.extract_vectors

# -------------------------------------------------------------
# 4. Train DPO model
# -------------------------------------------------------------
echo "Fine‑tuning with DPO..."
python -m src.train_dpo

# -------------------------------------------------------------
# 5. Evaluate aligned model
# -------------------------------------------------------------
echo "Evaluating aligned model..."
python -m src.evaluate --aligned

# -------------------------------------------------------------
# 6. Un‑align the model
# -------------------------------------------------------------
echo "Un‑aligning the model..."
python -m src.unalign

# -------------------------------------------------------------
# 7. Re‑evaluate after un‑alignment
# -------------------------------------------------------------
echo "Re‑evaluating after un‑alignment..."
python -m src.evaluate --unaligned

echo "Reproduction complete.  Results are in the 'results/' directory."