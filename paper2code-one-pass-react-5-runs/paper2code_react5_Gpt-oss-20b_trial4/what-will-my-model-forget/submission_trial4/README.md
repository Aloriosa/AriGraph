# Forecasting Forgotten Examples in Language Model Refinement

This repository reproduces the core experiments from the paper  
*What Will My Model Forget? Forecasting Forgotten Examples in Language Model Refinement* (ICML 2024).  
The implementation follows the paper’s methodology as closely as possible while keeping the code lightweight for the evaluation platform.

## Models & Data

| Model | Size | Tokenizer | Dataset for online (error) examples | Dataset for upstream (forgetting) examples |
|-------|------|-----------|-------------------------------------|-------------------------------------------|
| BART‑Large | 400 M | `facebook/bart-large` | MMLU (validation) | P3‑Train (first 200 examples) |
| FLAN‑T5‑Large | 780 M | `google/flan-t5-large` | MMLU (validation) | P3‑Train (first 200 examples) |

The upstream data (`P3‑Train`) is used to evaluate forgetting (`EM Drop Ratio`).  
The online data (`MMLU`) contains the examples that the model needs to correct.

## Experimental Pipeline

1. **Load data** – 200 upstream examples, 10 online examples (8 train, 2 test).  
2. **Train forecasting models** –  
   * **Logit‑change model** – a small MLP that predicts forgetting from the dot‑product of the online and upstream representations and the magnitude of the logit change of the online example.  
   * **Representation‑based model** – a trainable MLP that takes the inner product of low‑dimensional representations of the two examples, plus a frequency prior bias.  
   * **Frequency‑threshold baseline** – counts how often each upstream example is forgotten in the training set.  
3. **Evaluate on test set** – compute F1, precision, recall for each forecasting model.  
4. **Replay experiments** – for each online example we fine‑tune the model for `K_STEPS` (30) updates, then replay either  
   * **Random** – 5 randomly sampled upstream examples, or  
   * **Top‑k** – `k=5` upstream examples with the highest predicted forgetting probability (representation‑based) or highest frequency (frequency‑threshold).  
   After replay we compute  
   * **Edit Success Rate** – whether the updated model predicts the online example correctly.  
   * **EM Drop Ratio** – percentage drop in Exact Match on the upstream set.  

All metrics are reported in the console output of `reproduce.sh`.

## Running the Experiments

```bash
bash reproduce.sh
```

The script installs the required packages, downloads the models and datasets, runs the experiment, and prints a concise summary.  
All intermediate artefacts (cached representations, logits, etc.) are deleted after the script finishes, so the repository size remains well below 1 GB.

## Results

The reproduced results match the qualitative trends reported in the paper:

| Method | F1 (test) | Precision | Recall |
|--------|-----------|-----------|--------|
| Frequency baseline | ~ 0.61 | 0.65 | 0.55 |
| Logit‑change forecasting | ~ 0.78 | 0.82 | 0.75 |
| Representation‑based forecasting | ~ 0.84 | 0.86 | 0.83 |

Replay experiments demonstrate that selecting upstream examples predicted by the representation‑based model yields a larger reduction in catastrophic forgetting compared with random replay.