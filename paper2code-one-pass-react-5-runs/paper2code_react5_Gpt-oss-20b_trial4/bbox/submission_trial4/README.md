# BBox‑Adapter Reproduction

This repository contains a lightweight, fully reproducible implementation of the
**BBox‑Adapter** framework described in the paper  
“BBox‑ADAPTER: Lightweight Adapting for Black‑Box Large Language Models”.

> **Key features**  
> * Uses an open‑source GPT‑2 model as the “black‑box” LLM – no token probabilities are ever accessed.  
> * Implements a lightweight adapter based on DistilBERT + a linear head.  
> * Employs a ranking‑based Noise‑Contrastive Estimation (NCE) loss as described in §3.2.  
> * Provides an online adaptation loop that iteratively samples new candidates, constructs
>   positive/negative sets, and refines the adapter.  
> * Supports the four benchmark datasets (GSM8K, StrategyQA, TruthfulQA, ScienceQA)
>   via the 🤗 `datasets` library.  
> * Includes an end‑to‑end `reproduce.sh` script that installs dependencies,
>   runs training for each dataset, and prints test accuracy.  
> * No heavy checkpoints or proprietary APIs are committed – everything is generated on‑the‑fly.

> **Caveat**  
> The original paper targets state‑of‑the‑art black‑box LLMs (e.g., GPT‑3.5, Gemini) that do not expose token probabilities.  
> In this reproduction we use GPT‑2 as a stand‑in black‑box because it is freely available.  
> Consequently, the absolute numbers will differ from the paper, but the code demonstrates the full algorithmic pipeline.

## How to reproduce

```bash
# From the repository root
bash reproduce.sh
```

The script will:

1. Install the required Python packages.  
2. Train a BBox‑Adapter for each of the four datasets.  
3. Evaluate both the adapted model and a baseline that uses GPT‑2 alone.  
4. Print test accuracies for each setting.

All code is written in pure Python and PyTorch; it requires a CUDA‑capable GPU
to run efficiently, but will also run on CPU (slow).

## Repository structure

```
├── README.md
├── reproduce.sh
├── requirements.txt
├── train.py
├── adapter.py
├── prompt_templates.py
├── data_utils.py
├── ai_feedback.py
├── utils.py
└── .gitignore
```

Feel free to tweak hyper‑parameters in `train.py` or extend the prompt templates
to match the exact wording used in the paper.  The implementation follows
the spirit of the paper while staying lightweight and fully open‑source.