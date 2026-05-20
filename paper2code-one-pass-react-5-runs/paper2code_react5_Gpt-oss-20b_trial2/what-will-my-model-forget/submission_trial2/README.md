# Reproduction of “What Will My Model Forget? Forecasting Forgotten Examples in Language Model Refinement”

This repository contains a minimal, self‑contained implementation of the core ideas from the paper:

* Forecasting which upstream pre‑training examples will be forgotten when a language model is updated on an online error.
* A representation‑based forecasting model that predicts forgetting from learned sentence representations.
* A toy experiment using HuggingFace’s *BART‑large* and the *SQuAD* dataset to demonstrate the pipeline.

## How to run

```bash
bash reproduce.sh
```

The script will:

1. Load the datasets.
2. Evaluate the pre‑trained model on a small online error set and identify mis‑predicted examples.
3. Fine‑tune the model on each mis‑predicted example (10 gradient steps).
4. Determine which upstream examples are forgotten after each update.
5. Train a representation‑based forecasting model on the training split.
6. Evaluate the forecasting model on the test split and print:
   * Accuracy on the online error set (edit success rate)
   * EM drop ratio on the upstream examples
   * F1 / precision / recall of the forecasting model

All output is printed to stdout and written to `metrics.txt`.

> **Note**  
> The experiment uses only 100 examples from the SQuAD train split as upstream data and 100 examples from the validation split as the online error set, making it lightweight and fast (< 5 min on a single A10 GPU).

## Repository layout

```
/home/submission/
├─ requirements.txt          # Python dependencies
├─ README.md                 # This file
├─ reproduce.sh              # Reproduction driver
├─ main.py                   # Core implementation
└─ metrics.txt               # Generated after running reproduce.sh
```

The code intentionally focuses on the *core* algorithmic ideas rather than reproducing every detail of the original paper. It is fully deterministic and reproducible on the provided hardware configuration.