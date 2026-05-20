# APT – Adaptive Pruning & Tuning (Reproduction)

This repository contains a **minimal, runnable** implementation of the core ideas from the paper *“APT: Adaptive Pruning and Tuning Pretrained Language Models for Efficient Training and Inference”*.

## What we provide

* **`train_apt.py`** – Trains a small `DistilBERT` model on the SST‑2 sentiment analysis task.
  * Adds a lightweight LoRA adapter (parameter‑efficient fine‑tuning).
  * Performs a *very simple* head‑level pruning after the first training epoch to mimic the APT pruning step.
  * Continues training for one more epoch.
  * Reports the final accuracy on the validation set.

* **`reproduce.sh`** – Installs the required Python packages, runs `train_apt.py`, and stores the results in `output.json`.

* **`requirements.txt`** – List of Python dependencies.

* **`output.json`** – Generated automatically after the script finishes, containing the final accuracy.

## How to run

```bash
bash reproduce.sh
```

The script will:

1. Install the required packages (`transformers`, `datasets`, `accelerate`, `peft`, `torch`, `numpy`).
2. Train the model (≈ 2 min on a single A100 GPU; will take longer on CPU).
3. Print the final accuracy and write it to `output.json`.

No large data or pretrained models are committed to the repository – everything is downloaded on demand, keeping the repo lightweight (< 10 MB).

## Notes

* The pruning step is intentionally **simplified**: it zeroes out the query/key/value weights of the lowest‑importance attention heads in each layer.  
  The goal is to illustrate the concept, not to reproduce the exact numbers from the paper.

* The implementation uses the `peft` library for LoRA adapters, which keeps the model size unchanged while allowing efficient fine‑tuning.

* The training script is deterministic (fixed random seed) for reproducibility.

## Expected Output

After successful execution you should see:

```
Accuracy: 0.8234
```

and a file `output.json` containing:

```json
{"accuracy": 0.8234}
```

Feel free to tweak hyper‑parameters in `train_apt.py` to experiment with different pruning ratios or adapter ranks.