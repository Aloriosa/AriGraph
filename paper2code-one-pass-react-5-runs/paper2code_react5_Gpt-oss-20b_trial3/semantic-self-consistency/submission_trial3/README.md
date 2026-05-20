# Semantic Self‑Consistency Reproduction (Toy Version)

This repository contains a lightweight, end‑to‑end reproduction of the
semantic self‑consistency framework described in the paper
*“Semantic Self‑Consistency: Enhancing Language Model Reasoning via Semantic Weighting”*.

## What the repo contains

| File | Purpose |
|------|---------|
| `reproduce.sh` | Bash script that installs dependencies and runs the full pipeline. |
| `requirements.txt` | Python dependencies (PyTorch, HuggingFace Transformers, Sentence‑Transformers, scikit‑learn, etc.). |
| `main.py` | Implements the generation, featurisation, CPW, SCW, outlier filtering and evaluation. |
| `data/` | Three toy datasets (AQuA‑RAT, SVAMP, StrategyQA) in JSON‑Lines format. |
| `results/results.csv` | Output table of accuracies for each method, model and dataset. |

## How it works

1. **Generate chain‑of‑thought samples** – for each question we ask two open‑source LLMs (`gpt2` and `distilgpt2`) to produce 5 samples with a temperature of 0.7.
2. **Extract answers** – a very simple rule‑based extractor looks at the last line of the generation.
3. **Featurise** – the full rationale strings are passed through the
   `all-MiniLM-L6-v2` sentence‑transformer to obtain dense embeddings.
4. **Aggregation** – we apply:
   * Majority vote (self‑consistency baseline)
   * Centroid Proximity Weighting (CPW)
   * Semantic Consensus Weighting (SCW)
   * CPW/SCW after removing outliers with Isolation Forest
5. **Evaluation** – accuracy is computed by exact string match to the gold answer.
6. **Results** – a CSV table is written that can be inspected or plotted.

The toy datasets contain only 5 examples each, so the script finishes in a few seconds even on CPU.

## Running the reproduction

```bash
# Make the reproducibility script executable
chmod +x reproduce.sh

# Run the reproduction – this will build a Docker‑friendly environment
./reproduce.sh
```

After the script finishes you will find `results/results.csv` containing
the accuracy table.  The contents look like:

```
dataset,model,method,accuracy
AQuA‑RAT,gpt2,major,80.0
AQuA‑RAT,gpt2,cpw,80.0
...
```

Feel free to tweak `main.py` (e.g. change `n_samples`, add more models, or replace the extractor) – the structure follows the paper’s methodology closely.

## Limitations

- The datasets are tiny toy versions; results are illustrative only.
- The answer extraction is heuristic and may fail on more complex outputs.
- We use only two small open‑source LLMs; the paper used much larger models.
- The pipeline focuses on the semantic weighting algorithms (CPW, SCW) and
  simple outlier filtering; it does not cover the full breadth of the
  experiments in the paper.

Nevertheless, the code demonstrates how to combine generation,
embedding, weighting, outlier removal and evaluation in a reproducible
framework.