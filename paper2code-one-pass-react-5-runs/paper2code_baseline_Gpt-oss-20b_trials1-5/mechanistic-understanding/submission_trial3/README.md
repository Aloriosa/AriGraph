# Mechanistic Understanding of Alignment Algorithms – Reproduction

This repository reproduces a **simplified** version of the experiments described in  
*A Mechanistic Understanding of Alignment Algorithms: A Case Study on DPO and Toxicity*.  
The goal is to demonstrate the key ideas—training a toxicity probe, extracting toxic
value vectors, fine‑tuning a GPT‑2 model with Direct Preference Optimization (DPO),
and evaluating the resulting safety improvements—while staying lightweight enough
to run on a single NVIDIA A10 GPU within 7 days.

> **NOTE**: Because of resource constraints we use a toy implementation that
>  (1) trains a linear probe on the Jigsaw toxic comment dataset, (2) extracts
>  the most toxic value vectors from GPT‑2 medium, (3) fine‑tunes the model with
>  a simple DPO loss on a small Wikitext‑2 prompt set, and (4) evaluates toxicity
>  on a subset of the REALTOXICITYPROMPTS collection.  
>  The results are illustrative rather than identical to the paper.

## Repository Structure

```
.
├── README.md
├── reproduce.sh          # Main reproduction script
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── utils.py
│   ├── train_probe.py
│   ├── extract_toxic_vectors.py
│   ├── dpo_finetune.py
│   └── evaluate.py
└── output/               # Automatically created, contains models and logs
```

## How to Run

```bash
bash reproduce.sh
```

The script will:

1. Install the required Python packages.
2. Train a linear probe on the Jigsaw dataset and save the probe vector.
3. Extract the top‑10 toxic value vectors per GPT‑2 layer.
4. Fine‑tune the GPT‑2 medium base model with DPO on a small pairwise dataset.
5. Evaluate and print:
   * Toxicity probability (using the probe as a classifier).
   * Perplexity on a small Wikitext‑2 split.
   * “Un‑aligned” toxicity after amplifying key vectors.

All intermediate artefacts (model checkpoints, extracted vectors, logs)
are stored in the `output/` directory.

## Expected Outcome

Running the script should produce console output similar to:

```
[Probe] Training finished. Probe vector saved to output/probe/probe.pt
[Vectors] Extracted toxic vectors. Saved to output/vectors/
[DPO] Fine‑tuning finished. Best model saved to output/dpo/best_model.pt
[Eval] Baseline toxicity: 0.42 | DPO toxicity: 0.20 | Unaligned toxicity: 0.45
[Eval] Baseline PPL: 22.5 | DPO PPL: 23.0
```

The numbers are illustrative; the key point is that the DPO‑fine‑tuned model
shows a lower toxicity score while maintaining similar perplexity.

## Reproducibility

* All random seeds are fixed in `src/config.py`.
* Models are loaded from Hugging Face Transformers; no pre‑trained weights are
  stored in the repository to keep the size below 1 GB.
* The experiment uses a small subset of the datasets to keep runtime reasonable
  (≈ 15 minutes on an A10 GPU).

Feel free to experiment further by adjusting hyperparameters in
`src/config.py` or by replacing the probe with a more sophisticated
toxicity classifier.

---

*If you encounter any issues, please open an issue or contact the maintainer.*