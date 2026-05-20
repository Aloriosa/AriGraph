# Reproducing “What Will My Model Forget? Forecasting Forgotten Examples in Language Model Refinement”

This repository contains a lightweight, fully reproducible implementation of the core ideas from the paper *“What Will My Model Forget? Forecasting Forgotten Examples in Language Model Refinement”*.  
The goal is to illustrate how to:

1.  Fine‑tune a small pre‑trained language model (BERT‑Base) on a downstream task (SST‑2).  
2.  Collect *online* errors (validation examples that the model mis‑predicts).  
3.  Perform a *refinement* step for each online error and record which *training* examples become forgotten.  
4.  Train three forecasting methods (frequency‑threshold, logit‑change, representation‑based) that predict forgetting without re‑running the full model on all training data.  
5.  Evaluate the forecasting accuracy on a held‑out set of online errors.

The implementation is intentionally lightweight so that it runs on a single NVIDIA A10 GPU within a few minutes. It uses only the small BERT‑Base model and the SST‑2 dataset from the GLUE benchmark.

> **Note** – The numbers reported here are illustrative and *do not* match the large‑scale results published in the paper. The code is meant to be a faithful, runnable reproduction of the *methodology* rather than a full re‑run of the original experiments.

## Reproduction Script

Run the complete pipeline with:

```bash
bash reproduce.sh
```

The script will:

1.  Install the required packages.  
2.  Train the base model.  
3.  Perform the refinement and collect forgetting data.  
4.  Train and evaluate the forecasting models.  
5.  Write a `metrics.json` file in `outputs/` with F1 scores for each method.

After the script finishes, you should see:

```
Reproduction finished.
Results saved to outputs/metrics.json
```

Open `outputs/metrics.json` to inspect the metrics.

## Repository Structure

```
├── README.md
├── requirements.txt
├── reproduce.sh
├── outputs/
│   └── metrics.json          # Results
└── src/
    ├── data.py               # Load and cache datasets
    ├── train.py              # Fine‑tune base model
    ├── refine.py             # Perform refinement and collect forgetting
    ├── forecast_train.py     # Train forecasting models
    ├── forecast_eval.py      # Evaluate forecasting models
    └── utils.py              # Helper functions
```

## Expected Outputs

The `metrics.json` file will contain an entry for each forecasting method:

```json
{
  "threshold_f1": 0.48,
  "logit_f1": 0.55,
  "repr_f1": 0.61
}
```

These numbers are the F1 scores on the held‑out set of online errors.  
The script also prints a short summary to the console.

## License

This code is released under the MIT license.  
It is provided for educational and research purposes only.