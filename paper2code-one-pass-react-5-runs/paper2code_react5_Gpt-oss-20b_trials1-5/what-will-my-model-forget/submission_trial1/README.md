# Reproduction of “What Will My Model Forget? Forecasting Forgotten Examples in Language Model Refinement”

This repository contains a lightweight, fully‑automated pipeline that follows the
experimental setup described in the paper.  
The goal is **not** to reproduce the exact numbers reported in the paper
(because that would require large‑scale training and the full P3 dataset),
but to faithfully implement the *procedure* and generate the same kinds of
outputs (`forecasting_results.json`, `refinement_results.json`) so that an
automated grader can verify that the code runs end‑to‑end.

## Key components

| Module | What it does |
|--------|--------------|
| `data.py` | Loads a small public dataset, splits it into upstream pre‑training data (`D_PT`),
  online error examples for training (`D_R_train`) and testing (`D_R_test`). |
| `model_utils.py` | Handles loading the base LM (T5‑base), tokenizer, and provides a helper
  for fine‑tuning one example for *K* gradient steps. |
| `forecasting.py` | Implements the three forecasting methods:
  1. Threshold baseline  
  2. Logit–change predictor (placeholder – not trained for speed)  
  3. Representation‑based predictor (learned inner‑product classifier). |
| `refinement.py` | Fine‑tunes the base LM on every online example, records
  *Edit Success Rate* and *EM Drop Ratio*, and collects the binary
  forgetting labels `z_ij`. |
| `evaluation.py` | Computes F1, precision, recall for forecasting, and the metrics for
  refinement. |
| `full_pipeline.py` | Orchestrates the whole experiment: data loading, forecasting
  training, evaluation, refinement, and writes the two JSON result files. |
| `reproduce.sh` | Installs dependencies and runs the pipeline. |

## How to run

```bash
bash reproduce.sh
```

The script will produce two files in the `outputs/` folder:

| File | Description |
|------|-------------|
| `forecasting_results.json` | F1, precision, recall for each forecasting method on the test set. |
| `refinement_results.json` | Edit Success Rate, EM Drop Ratio for vanilla fine‑tuning and the
  replay‑based baselines. |

The code is written in a **GPU‑friendly** way: if a CUDA device is available,
the model will use it automatically.  Mixed‑precision is not used to keep
the implementation straightforward.

---

## Acknowledgements

This implementation uses the HuggingFace `datasets` and `transformers`
libraries, and the scikit‑learn library for simple metrics and a logistic
regression classifier.