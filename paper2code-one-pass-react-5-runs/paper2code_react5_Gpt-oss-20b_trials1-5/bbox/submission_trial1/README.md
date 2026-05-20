# BBox‑Adapter – Lightweight Black‑Box LLM Adaptation

This repository contains a minimal, self‑contained implementation of the **BBox‑Adapter** method described in the paper *“BBox‑ADAPTER: Lightweight Adapting for Black‑Box Large Language Models”*.  
The goal is to provide a reproducible pipeline that:

1. **Trains** a small adapter that scores candidate responses from a black‑box language model.  
2. **Evaluates** the adapter on a subset of the GSM8K math‑word‑problem dataset.  
3. **Runs** entirely on a local GPU or CPU without requiring any external API keys.

> **Note** – This is a *toy* version that reproduces the core idea (ranking‑based NCE loss and online adaptation loop) on a small scale.  
> The full experimental protocol in the paper requires far larger models, datasets, and more sophisticated training schedules, which are beyond the scope of this repository.

## Repository Structure

```
├── adapters.py          # Lightweight adapter definition
├── train.py             # Training script (online adaptation + NCE loss)
├── evaluate.py          # Evaluation script
├── utils.py             # Helper functions (generation, batching, loss)
├── reproduce.sh         # End‑to‑end reproduction script
├── requirements.txt     # Dependencies
└── README.md            # Documentation
```

## Reproducing the Results

```bash
bash reproduce.sh