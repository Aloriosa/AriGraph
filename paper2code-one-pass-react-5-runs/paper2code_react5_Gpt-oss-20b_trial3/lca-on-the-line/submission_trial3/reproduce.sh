#!/usr/bin/env bash
set -e

# ------------------------------------------------------------------
# 1. Install the required Python packages
# ------------------------------------------------------------------
python3 -m pip install --upgrade pip
pip install -r requirements.txt

# ------------------------------------------------------------------
# 2. Download the ImageNet validation set and the five OOD datasets.
#    The datasets are downloaded automatically by the evaluation script.
# ------------------------------------------------------------------
python3 eval_models.py --download-only

# ------------------------------------------------------------------
# 3. Run the full evaluation (ID + OOD + LCA + correlation)
# ------------------------------------------------------------------
python3 eval_models.py --run-eval

# ------------------------------------------------------------------
# 4. The script will generate the following artefacts:
#     - results/id_accuracies.csv
#     - results/ood_accuracies.csv
#     - results/lca_scores.csv
#     - results/correlation_results.csv
#     - results/plots/correlation.png
# ------------------------------------------------------------------
echo "Reproduction finished. Please see the 'results/' directory for output."