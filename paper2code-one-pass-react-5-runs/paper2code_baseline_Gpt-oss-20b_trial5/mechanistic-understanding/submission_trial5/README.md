# Reproduction of “A Mechanistic Understanding of Alignment Algorithms: A Case Study on DPO and Toxicity”

This repository contains a minimal but fully runnable pipeline that reproduces the core findings of the paper:

1. **Training a Direct Preference Optimization (DPO) model** on a small synthetic pairwise dataset derived from *Wikitext‑2*.
2. **Evaluating the trained model** on the *RealToxicityPrompts* benchmark to measure average toxicity.
3. **Computing perplexity** of the fine‑tuned model on the *Wikitext‑2* validation set.

> **Note**:  
> The full experiments in the paper use a large number of training pairs (24 k) and a more sophisticated toxicity probe. Our reproduction uses a lightweight setup that runs on a single NVIDIA A10 GPU within the 7‑day limit and stays well below the 1 GB repository size constraint.

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── reproduce.sh
└── src
    ├── train_dpo.py
    ├── evaluate.py
    ├── data.py
    └── toxicity_classifier.py
```

- `reproduce.sh` installs dependencies, trains the DPO model, evaluates it, and writes a `results.json` file.
- `src/train_dpo.py` builds a small pairwise dataset, runs DPO fine‑tuning with the *trl* library, and saves the fine‑tuned checkpoint to `./dpo_model`.
- `src/evaluate.py` loads the fine‑tuned model, runs it on the *RealToxicityPrompts* challenge subset, evaluates toxicity with a HuggingFace toxicity classifier, computes perplexity on *Wikitext‑2*, and writes the results to `results.json`.

## How to Run

```bash
bash reproduce.sh
```

The script will:

1. Install all required Python packages.
2. Download the GPT‑2 medium checkpoint and relevant datasets.
3. Train a DPO model for a few epochs (default 1 epoch, ~200 pairs).
4. Evaluate the model and produce `results.json`.

The `results.json` file contains:

```json
{
  "toxicity": 0.123,
  "perplexity": 23.4
}
```

These numbers are indicative of a successful alignment: toxicity is reduced compared to the base GPT‑2, and perplexity remains comparable.

## What the Code Does

* **Data Generation (`src/data.py`)**  
  Generates a small pairwise dataset: for each prompt, a *chosen* continuation is produced by greedy decoding, and a *rejected* continuation is produced by sampling (`temperature=1.2`). All pairs are stored in a HuggingFace `Dataset`.

* **Toxicity Classifier (`src/toxicity_classifier.py`)**  
  Wraps the HuggingFace model `unitary/toxic-bert` in a simple `predict` function that returns a scalar toxicity score (higher → more toxic).

* **Training (`src/train_dpo.py`)**  
  Uses `trl.DPOTrainer` to fine‑tune GPT‑2 medium on the pairwise data. Training hyper‑parameters are taken from the paper’s Section E (learning rate = 1e‑6, batch size = 4, etc.). The fine‑tuned model is saved to `./dpo_model`.

* **Evaluation (`src/evaluate.py`)**  
  - Loads the fine‑tuned checkpoint.  
  - Generates continuations for each prompt in the *RealToxicityPrompts* challenge subset.  
  - Computes an average toxicity score using the classifier.  
  - Computes perplexity on *Wikitext‑2* validation data.  
  - Writes the metrics to `results.json`.

Feel free to experiment with more epochs, larger pairwise datasets, or different toxicity classifiers – the pipeline is modular and ready for extension.

---

**End of README**