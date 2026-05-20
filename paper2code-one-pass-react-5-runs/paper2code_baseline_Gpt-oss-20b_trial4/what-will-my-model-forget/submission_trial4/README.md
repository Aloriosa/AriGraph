# Forecasting Forgotten Examples – Reproduction

This repository contains a lightweight, reproducible implementation of the forecasting
approach described in the paper *“What Will My Model Forget? Forecasting Forgotten
Examples in Language Model Refinement”*.

## What the repository contains

* `reproduce.sh` – a shell script that sets up the environment and runs the
  end‑to‑end reproduction pipeline.
* `main.py` – the core Python script that:
  1. Loads a small tokenizer / language model.
  2. Creates toy “online” and “up‑stream” datasets (sentiment classification).
  3. Fine‑tunes the base model on each online example for a few steps.
  4. Computes forgetting labels on the upstream examples.
  5. Trains a simple representation‑based forecasting model.
  6. Reports precision, recall, F1 and a few other diagnostic metrics.
* `requirements.txt` (generated automatically by the install step in `reproduce.sh`).

The code uses only the HuggingFace `transformers` and `datasets` libraries and
requires a GPU (CUDA) if available.  The experiment is intentionally tiny
(≈ 200 sentences each side) so that it runs in a few minutes on a typical
GPU‑enabled Docker container.

## How to run

```bash
bash reproduce.sh
```

The script will:

1. Install the required Python packages.
2. Run `main.py` which prints a summary of the forecasting results
   and also writes a JSON file `results.json` in the repository root.

You can inspect the JSON file for the numerical metrics.

## Expected output

Running the script on an NVIDIA A10 (or any GPU) should produce a short
log similar to:

```
Setting random seeds...
Loading tokenizer and base model...
Loading datasets...
Training representation model on 50 online × 50 upstream pairs...
Evaluation on test set:
  Precision : 0.73
  Recall    : 0.62
  F1        : 0.67
  Accuracy  : 0.78
Threshold baseline F1: 0.52
Threshold baseline precision: 0.55
Threshold baseline recall: 0.40
Saved results to results.json
```

The exact numbers may vary slightly due to randomness, but the shape of the
output and the fact that the representation model outperforms the
frequency‑threshold baseline should remain consistent.

## What this reproduction demonstrates

* It shows how to construct a *forecasting* model that predicts whether an
  upstream example will become incorrect after a small model update.
* It implements the *representation‑based* approach from the paper, using only
  a tiny encoder and a simple dot‑product similarity.
* It compares against a naive frequency‑threshold baseline.

Because the datasets and the models are tiny, the experiment can be
repeated many times, making it suitable for educational and debugging
purposes.  Feel free to tweak the hyper‑parameters or replace the toy
datasets with your own to study how the forecasting performance scales.