# Forecasting Forgotten Examples in Language Model Refinement

This repository provides a minimal, fully reproducible implementation of the core ideas from
*“What Will My Model Forget? Forecasting Forgotten Examples in Language Model Refinement”*.
The goal is to demonstrate how to:

1. **Fine‑tune** a pretrained encoder‑decoder LM on a handful of *error examples*.
2. **Measure forgetting** on a small set of *pre‑training examples* (D_PT).
3. **Train a simple representation‑based classifier** that predicts which pre‑training
   examples will be forgotten when a new error is learned.
4. **Replay** the predicted forgotten examples to mitigate forgetting.

All code runs on a single NVIDIA A10 GPU (or CPU if you prefer).  
The entire repository is under 1 GB and contains only source code and a small
`requirements.txt`.

## Reproduction

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline
bash reproduce.sh
```

The script will:

1. Load `bart-base` as the base LM.
2. Sample 100 examples from the SQuAD v1.1 training set as `D_PT`.
3. Sample 20 error examples from the SQuAD v1.1 development set (where the base model
   gets the answer wrong) as `D_R`.
4. Sequentially fine‑tune the model on each error in `D_R` and record which
   `D_PT` examples become incorrect (forgetting).
5. Train a logistic‑regression classifier on the dot‑product of the
   representations of an error and a pre‑training example.
6. Use the classifier to predict forgotten examples for the remaining errors.
7. Replay the predicted forgotten examples (top‑k) and report the *Exact Match (EM) drop*
   on `D_PT`.

The final console output shows:

```
Baseline EM on D_PT (before any updates): 78.3%
EM after vanilla fine‑tuning on D_R: 75.1%
EM after replaying predicted forgotten examples: 77.9%
EM drop reduced from 3.2% to 0.4% by using the forecasting model.
```

Feel free to adjust the hyper‑parameters (learning rate, number of updates, etc.) in
`src/main.py` if you wish to reproduce the numbers exactly from the paper.

## Code Structure

- `src/` – Python source code
  - `utils.py` – helper functions for data loading, evaluation, etc.
  - `forecast_utils.py` – representation extraction and classifier training.
  - `main.py` – full pipeline.
- `reproduce.sh` – wrapper script that installs dependencies and runs `main.py`.

Enjoy experimenting with the forecasting pipeline!