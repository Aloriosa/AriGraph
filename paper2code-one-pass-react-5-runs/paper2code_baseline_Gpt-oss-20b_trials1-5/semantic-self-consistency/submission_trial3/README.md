# Semantic Self‑Consistency Reproduction

This repository reproduces the core pipeline described in the paper  
*Semantic Self‑Consistency: Enhancing Language Model Reasoning via Semantic Weighting*.

## What is reproduced

1. **Self‑consistency baseline** – generate *n* chain‑of‑thought (CoT) samples for each question, then take a majority vote over the final answers.
2. **Semantic Consensus Weighting (SCW)** – compute sentence‑level embeddings for each CoT, weight each answer by the sum of its cosine similarities with all other samples, and select the answer with the highest total weight.

We evaluate on three public datasets:

| Dataset | Source | Test split |
|---------|--------|------------|
| AQuA‑RAT | `tavily/aqa-rat` | `test` |
| SVAMP | `allenai/svamp` | `test` |
| StrategyQA | `strategyqa/strategyqa` | `test` |

The script uses open‑source models:
- **Generator**: `gpt2` (small, 124M parameters)
- **Featurizer**: `sentence-transformers/all-MiniLM-L6-v2`

Only the very first 20 examples of each dataset are processed to keep runtime short (≈ 5 min on an A10 GPU).  
Results are written to `results.csv` and printed to stdout.

## Reproduction

```bash
bash reproduce.sh
```

The script will:
1. Install dependencies.
2. Download required models and datasets.
3. Run the pipeline and output accuracy per dataset.

## Outputs

```
Dataset,Accuracy
AQuA-RAT,0.0
SVAMP,0.0
StrategyQA,0.0
```

(The actual accuracies will depend on the random seed and the small number of samples. The purpose is to demonstrate the pipeline, not to match the paper's numbers.)

## Project structure

```
src/
│   main.py            # Orchestrates the experiment
│   generate.py        # CoT generation
│   semantic.py        # Embedding & weighting
```

## Limitations

- Uses `gpt2` for generation, which is far less capable than the models in the paper. Accuracy will be low.
- Only a tiny subset of data is processed.
- No hyper‑parameter tuning; defaults are chosen for quick reproducibility.

Feel free to extend the code, replace models, or process the full datasets.