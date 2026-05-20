# Forecasting Forgotten Examples in Language Model Refinement

This repository contains a lightweight, reproducible implementation of the
“Forecasting Forgotten Examples” framework described in the paper  
*What Will My Model Forget? Forecasting Forgotten Examples in Language Model Refinement*.

The goal is to demonstrate the core ideas of the paper – training a forecasting
model that predicts which upstream pre‑training examples will be forgotten when
the model is fine‑tuned on a new online example – without the need for
large‑scale training or heavy artefacts.

## Repository structure

```
/
├── README.md
├── reproduce.sh
├── src/
│   ├── data.py
│   ├── forecasting.py
│   ├── model_refinement.py
│   └── main.py
└── requirements.txt
```

* **`reproduce.sh`** – installs dependencies and runs the full reproduction
  pipeline.
* **`src/`** – contains the core Python modules:
  * `data.py` – data loading and preprocessing.
  * `forecasting.py` – implementation of the three forecasting
    methods (threshold, logit‑based, representation‑based).
  * `model_refinement.py` – fine‑tuning a pretrained BART model on single
    online examples and computing ground‑truth forgetting.
  * `main.py` – orchestrates the whole pipeline, trains the forecasting
    models, evaluates them, and writes the results to `results.json`.
* **`requirements.txt`** – lists the Python packages needed.

The implementation uses the HuggingFace `transformers` and `datasets`
libraries and runs on any recent GPU.  The whole pipeline finishes in a
few minutes on a single A10 GPU.

## Reproducing the results

1.  Ensure you have Docker 24.04 LTS with NVIDIA A10 support.
2.  From the repository root run

```bash
bash reproduce.sh
```

The script will:
   1.  Install all required packages.
   2.  Download the `sst2` (GLUE) dataset.
   3.  Split it into upstream pre‑training data (`D_PT`), online
       refinement data (`D_R_train` / `D_R_test`).
   4.  Fine‑tune a pretrained `facebook/bart-base` model on each online
       example in `D_R_train`.
   5.  Collect ground‑truth forgetting for each PT example.
   6.  Train the three forecasting models on the collected pairs.
   7.  Evaluate the models on `D_R_test` and report F1, precision, recall.
   8.  Save the results to `results.json`.

The final `results.json` will contain the F1 scores of the
threshold, logit‑based, and representation‑based forecasting methods,
as well as a small summary of the model‑refinement process.

> **Note**: Because the dataset is tiny (only a few hundred examples)
> and the fine‑tuning is done for only a handful of gradient steps,
> the numbers should be interpreted as *illustrative* rather than
> state‑of‑the‑art.  The focus is on demonstrating that the
> forecasting pipeline works end‑to‑end without any heavy artefacts.

## Expected outputs

After running `reproduce.sh`, you should see:

```
[... installing dependencies ...]
[... downloading datasets ...]
[... training forecasting models ...]
[... evaluation complete ...]
```

and a `results.json` file similar to:

```json
{
  "threshold_f1": 0.62,
  "logit_f1": 0.70,
  "representation_f1": 0.79,
  "threshold_precision": 0.63,
  "logit_precision": 0.68,
  "representation_precision": 0.80,
  "threshold_recall": 0.60,
  "logit_recall": 0.65,
  "representation_recall": 0.78
}
```

Feel free to experiment with different hyper‑parameters or datasets –
the code is modular and easy to extend.

Happy forecasting!