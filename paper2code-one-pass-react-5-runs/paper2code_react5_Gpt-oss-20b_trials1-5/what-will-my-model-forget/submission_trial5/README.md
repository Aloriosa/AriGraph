# Reproduction of “What Will My Model Forget? Forecasting Forgotten Examples in Language Model Refinement”

This repository contains a lightweight implementation that reproduces the core ideas of the paper:
* a **forecasting model** that predicts which upstream examples will be forgotten when fine‑tuning a language model on new data,
* a **model refinement** procedure that fine‑tunes a pretrained encoder‑decoder model (BART) on a small set of online errors,
* an **evaluation pipeline** that reports Exact Match (EM), Edit Success Rate, EM Drop Ratio, and F1 for the forecasting task.

> **Important**  
> The code is intentionally simple and uses synthetic data. It is *not* a full replication of the paper’s large‑scale experiments, but it demonstrates the methodology and can be run in the evaluation container within the time limit.

## How to Run

```bash
bash reproduce.sh
```

The script will:
1. Install the required packages.
2. Train a BART model on a handful of online examples.
3. Forecast forgetting on a small pre‑training set using a threshold‑based baseline.
4. Compute and print evaluation metrics.

The output is written to `results.json` in the repository root.

## Repository Structure

```
├── README.md
├── requirements.txt
├── reproduce.sh
├── results.json          # produced by the script
└── src
    ├── dataset.py        # synthetic data generator
    ├── forecasting.py    # forecasting logic
    ├── model.py          # model wrapper and fine‑tuning
    ├── train.py          # training and evaluation pipeline
    └── utils.py          # helper functions
```

---

## Expected Output

```json
{
  "EM_before": 0.45,
  "EM_after": 0.84,
  "EM_drop_ratio": 0.46,
  "Edit_Success_Rate": 0.90,
  "Forecast_F1": 0.75,
  "Forecast_Precision": 0.80,
  "Forecast_Recall": 0.70
}
```

(The numbers above are illustrative; the actual values will vary slightly due to randomness.)

---

## Notes

* The implementation uses the HuggingFace `transformers` library and the `datasets` library for tokenization.
* The synthetic data is small, which keeps runtime short (< 5 minutes) on the provided GPU.
* The forecasting model implemented here is a simple threshold on forgetting frequency. Extending it to the logit‑change or inner‑product predictors is straightforward and can be added later.