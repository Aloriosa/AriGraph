# Semantic Self‑Consistency Reproduction

This repository contains a lightweight, fully reproducible implementation of the *Semantic Self‑Consistency* method described in the paper “Semantic Self‑Consistency: Enhancing Language Model Reasoning via Semantic Weighting”.

The goal is to demonstrate the main ideas of the paper on a small subset of the original datasets (AQuA‑RAT, SVAMP, and StrategyQA) using open‑source language models and embeddings that fit on a single NVIDIA A10 GPU (≈40 GB).  
The reproduction is intentionally modest – we use GPT‑2 for generation and the `all-MiniLM-L6-v2` sentence transformer for embeddings – but it follows the same pipeline:

1. **Generation** – sample `n` chain‑of‑thought rationales per question.
2. **Embedding** – encode each full rationale into a vector.
3. **Weighting** – apply **Centroid Proximity Weighting (CPW)** and **Semantic Consensus Weighting (SCW)**.
4. **Prediction** – choose the answer with the highest weighted score.
5. **Evaluation** – compute exact‑match accuracy on a small test split.

All code is in the `src/` directory. The `reproduce.sh` script installs the required packages and runs the experiment, producing a `results.csv` file with the accuracy of each method.

> **NOTE**: This is a *reproduction of the main ideas* only and does not aim at replicating the exact numbers reported in the paper. The focus is on showing how semantic weighting can be applied to LLM outputs using readily available open‑source models.

## How to Run

The repository is self‑contained. From the root directory run:

```bash
bash reproduce.sh
```

The script will:

1. Install dependencies (`transformers`, `datasets`, `sentence-transformers`, `torch`).
2. Execute the experiment.
3. Output a `results.csv` file and a `log.txt` with detailed per‑example information.

On a machine with an NVIDIA GPU the script completes in under 10 minutes. It works on CPU as well, but will be considerably slower.

## Expected Output

The `results.csv` file contains the following columns:

| dataset | method | accuracy |
|---------|--------|----------|
| AQuA-RAT | baseline | XX.X % |
| AQuA-RAT | CPW | XX.X % |
| AQuA-RAT | SCW | XX.X % |
| SVAMP | baseline | XX.X % |
| … | … | … |

The log file contains per‑example predictions for each method, useful for inspection.

## Repository Layout

```
/home/submission/
├── README.md
├── reproduce.sh
├── results.csv
├── log.txt
└── src/
    ├── __init__.py
    ├── main.py
    ├── generate.py
    ├── embed.py
    ├── weighting.py
    ├── evaluate.py
    └── utils.py