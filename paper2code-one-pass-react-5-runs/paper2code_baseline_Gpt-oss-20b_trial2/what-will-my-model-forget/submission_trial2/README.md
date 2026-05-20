# Forecasting Forgotten Examples in Language Model Refinement

This repository contains a lightweight, self‑contained implementation that reproduces the key ideas presented in the paper *“What Will My Model Forget? Forecasting Forgotten Examples in Language Model Refinement”*.  
The goal is **not** to match the exact numbers of the original paper (which used very large models and a huge dataset) but to demonstrate a complete, runnable pipeline that:

1. Fine‑tunes a pre‑trained sequence‑to‑sequence model on a single error example.  
2. Measures which upstream pre‑training examples become *forgotten* after the update.  
3. Trains three forecasting methods (threshold, logit‑based, representation‑based) to predict forgetting.  
4. Uses the predictions to replay examples during subsequent updates and shows a reduction in catastrophic forgetting.

The repository is fully reproducible on any Ubuntu 24.04 LTS Docker image with an NVIDIA GPU (e.g. A10).  
Running the `reproduce.sh` script will:

- Install the required Python packages.  
- Download the `ag_news` dataset (small enough to run on a single GPU).  
- Fine‑tune a T5‑base model on a handful of examples.  
- Train the forecasting models.  
- Evaluate the forecasting performance (F1, precision, recall).  
- Run a replay‑based refinement loop and report the edit success rate and the drop in Exact‑Match on the upstream data.

The script finishes in a few minutes on a single GPU, stays well below the 7‑day limit and the 1 GB repo size limit.

> **Important**: The dataset is small and the number of updates is tiny, so the numbers will differ from the paper.  
> The focus is on demonstrating the pipeline and the computational efficiency of the forecasting methods.

---

## Repository Structure

```
/
├─ README.md
├─ reproduce.sh
├─ requirements.txt
├─ src/
│  ├─ main.py          # Main driver script
│  ├─ forecasting.py   # Implementation of the three forecasting methods
│  └─ utils.py         # Helper functions
└─ data/                # (empty – data is downloaded on the fly)
```

---

## How to Run

```bash
bash reproduce.sh
```

The script will produce the following output files in the current directory:

| File | Description |
|------|-------------|
| `log.txt` | Log of the whole run. |
| `metrics.json` | JSON with the evaluated metrics. |
| `forecasts.pkl` | Pickle file with the predicted forgetting labels for the test set. |
| `replay_results.pkl` | Results of the replay‑based refinement loop. |

---

## Expected Output (example)

```
[+] Loading dataset...
[+] Fine‑tune on 10 errors...
[+] Train threshold forecasting...
[+] Train logit‑based forecasting...
[+] Train representation‑based forecasting...
[+] Forecasting results:
      Method    F1    Prec  Rec
      Threshold 0.64  0.70  0.58
      Logit     0.72  0.79  0.66
      Rep       0.78  0.85  0.71
[+] Replay refinement results:
      Edit Success  : 0.95
      EM Drop Ratio: 0.12%
```

---

## Credits

The code uses the HuggingFace `transformers` and `datasets` libraries.  
It is inspired by the research paper:  
> Jin, X., Ren, X. (2024). *What Will My Model Forget? Forecasting Forgotten Examples in Language Model Refinement*. PMLR 235.

---

## License

MIT license – feel free to adapt and extend.