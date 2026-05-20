# Forecasting Forgotten Examples in Language Model Refinement

This repository contains a **minimal, runnable** implementation that reproduces the core ideas from the paper:

*Train a simple representation‑based forgetting predictor, use it to select examples for replay, and measure the impact on catastrophic forgetting.*

> The full experiments in the paper involve large models (BART‑Large, FLAN‑T5‑Large/3B) and thousands of examples.  
> Here we use lightweight models (`facebook/bart-base`, `google/flan-t5-base`) and a tiny data subset
> so that the entire pipeline runs in a few minutes on a single GPU.

## Repository layout

| File | Purpose |
|------|---------|
| `reproduce.sh` | Shell script that installs dependencies, downloads models, and runs the experiment. |
| `requirements.txt` | Python dependencies (torch, transformers, datasets, etc.). |
| `run_experiment.py` | Main script that: 1) loads data, 2) trains a representation‑based forecaster, 3) evaluates it, 4) runs replay‑based refinement, 5) outputs metrics. |
| `forecasting.py` | Optional helper module (same model as in `run_experiment.py`). |
| `utils.py` | Helper functions for tokenisation and representation extraction. |
| `README.md` | This documentation. |

## Reproduction steps

1. **Clone the repo** (or simply copy the files to a local directory).  
2. **Run the reproduction script**:

   ```bash
   bash reproduce.sh
   ```

   *The script will:*
   * install Python dependencies,
   * download the base models (`bart-base` and `flan-t5-base`),
   * run the experiment, and
   * write a `results.json` file with the final metrics.

3. **Check the output**:

   ```bash
   cat results.json
   ```

   Expected fields (values will vary slightly due to randomness):

   ```json
   {
     "forecaster": {
       "f1": 0.55,
       "precision": 0.60,
       "recall": 0.45
     },
     "replay": {
       "edit_success_rate": 92.5,
       "em_drop_ratio": 2.1
     }
   }
   ```

   *The numbers are illustrative; the key point is that the script runs end‑to‑end.*

## What the experiment does

1. **Data preparation**  
   * Upstream data: 200 sentences from the *wikitext* dataset (used as “pre‑training” examples).  
   * Online learning data: 10 examples from the *SQuAD* train split (used as “errors” to be fixed).

2. **Forecaster training**  
   * For each online example, the model is fine‑tuned for 1 epoch.  
   * The change in predictions on all upstream examples is recorded (whether the example is forgotten).  
   * A simple 2‑layer MLP encodes the pooled representation of each example; the dot product of two such vectors predicts forgetting.

3. **Evaluation**  
   * F1, precision and recall of the forecaster on a held‑out online example.

4. **Replay‑based refinement**  
   * For each online example, after fine‑tuning, we identify upstream examples that the forecaster predicts will be forgotten.  
   * The model is further fine‑tuned on those examples (replay).  
   * We measure **Edit Success Rate** (how often the online example is correctly answered after the update) and **EM Drop Ratio** (percent drop in accuracy on upstream examples).

## Limitations

* This is a **toy implementation**: the data split is tiny, the models are small, and the training loop is heavily simplified.
* The forgetting predictor uses a *very* simple architecture and a single forward pass per example; it is not meant to match the performance reported in the paper.
* The goal is to provide an end‑to‑end runnable pipeline that demonstrates the *mechanism* of forecasting forgetting and using it for replay.

## Customization

Feel free to modify the following parameters in `run_experiment.py`:

| Variable | Description | Default |
|----------|-------------|---------|
| `UPSTREAM_SPLIT` | Number of upstream examples | `wikitext-2-raw-v1.train[:200]` |
| `REFINEMENT_NUM_EXAMPLES` | Number of online learning examples | `10` |
| `NUM_REPLAY` | How many upstream examples to replay per online update | `4` |

You can also swap in different HuggingFace models by changing the `BASE_MODELS` dictionary.

## Contact

If you have questions, feedback, or want to extend the toy experiment, feel free to open an issue or reach out to the corresponding author of the paper.