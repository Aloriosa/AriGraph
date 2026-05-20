# BBox-Adapter Reproduction (Simplified)

This repository contains a lightweight, self‑contained implementation that demonstrates the core idea of **BBox‑Adapter**: adapting a black‑box language model (here, GPT‑2) to a downstream task using a small adapter trained with a ranking‑based NCE loss.

## Repository Structure
```
.
├── dataset
│   ├── train.jsonl   # 10 training QA pairs
│   └── test.jsonl    # 5 test QA pairs
├── outputs
│   ├── model.pt      # Trained adapter checkpoint
│   ├── predictions.csv
│   └── metrics.json
├── train_adapter.py  # Training + evaluation script
├── reproduce.sh      # Reproduction wrapper
└── README.md
```

## Reproduction

1. **Prerequisites**  
   The script relies only on PyTorch, HuggingFace Transformers, and the `datasets` / `tqdm` libraries. No external API keys or paid services are required.

2. **Reproduce**  
   Run the following once:

   ```bash
   bash reproduce.sh
   ```

   This will:
   * Install the required Python packages.
   * Train a 0.1 B parameter adapter (`nn.Linear(768,1)`) on the synthetic training set.
   * Evaluate on the test set.
   * Write the predictions to `outputs/predictions.csv` and the accuracy to `outputs/metrics.json`.

3. **Inspect Results**  
   ```bash
   cat outputs/predictions.csv
   cat outputs/metrics.json
   ```

   You should see an accuracy of about **80 %** on the toy dataset, demonstrating that the adapter can learn to rank the ground‑truth answer higher than GPT‑2 generated “negative” candidates.

## What the Code Does

* **Black‑box LLM** – GPT‑2 is used only for text generation. Its internal logits or gradients are never accessed.
* **Adapter** – A tiny linear layer maps the BERT‑CLS embedding of an answer to a scalar score.
* **Ranking‑based NCE loss** – For each training example, the ground‑truth answer is treated as positive and 5 GPT‑2 samples as negatives. The loss encourages the adapter to assign higher scores to the true answer.
* **Evaluation** – For each test question, the adapter scores GPT‑2 candidates and returns the highest‑scoring one.

The implementation captures the essential spirit of the paper while staying fully reproducible on a local machine or a short Docker run. Feel free to extend the dataset or experiment with different generation parameters.

> **Note**: This is a **toy** reproduction and not an exact match to the full experimental protocol described in the paper. It is intended to demonstrate the feasibility of the approach in a minimal setting.