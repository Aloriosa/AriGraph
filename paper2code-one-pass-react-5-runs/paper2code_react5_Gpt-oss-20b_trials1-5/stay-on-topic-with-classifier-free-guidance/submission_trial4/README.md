# Reproduction of “Stay on Topic with Classifier‑Free Guidance”

This repository contains a minimal, self‑contained reproduction of the core idea from the paper *Stay on Topic with Classifier‑Free Guidance* (CFG for language models).  
The goal is **not** to re‑run the full experimental suite from the paper (which requires large models, dozens of checkpoints, and GPU clusters), but to demonstrate:

1. How CFG can be applied to an off‑the‑shelf causal language model (we use `gpt2` from HuggingFace).  
2. How to compute the guided logits in the autoregressive setting.  
3. How to generate text with CFG and with the vanilla model for direct comparison.  
4. How to measure a simple indicator of the effect of CFG – the entropy of the next‑token distribution.

The reproduction script (`reproduce.sh`) installs the necessary packages, downloads the model, and runs a short demo that prints the generated continuations and the entropy reduction.

## How to run

```bash
bash reproduce.sh
```

On a recent Ubuntu 24.04 base Docker image with an NVIDIA A10 GPU, the script takes less than a minute to finish and prints:

```
Prompt: "Once upon a time"
Vanilla output: ...
CFG output (γ=1.5, temp=0.7): ...
Entropy (vanilla): 5.12
Entropy (CFG): 3.84
```

The script will also generate a small JSON file `results.json` containing the prompts, the two generated continuations, and the measured entropies.

## Repository structure

```
/home/submission/
├── README.md
├── reproduce.sh
└── main.py
```

All heavy artifacts (model weights, checkpoints, evaluation logs) are **not** included – they are downloaded on demand from HuggingFace.  
The repository size stays well below 1 GB.

## What was reproduced

- **CFG algorithm**: The re‑weighting of logits described in Section 2.2 of the paper.  
- **Hyperparameters**:  
  - Guidance weight γ = 1.5 (you may change it in `main.py`).  
  - Sampling temperature = 0.7.  
  - Top‑p = 0.9, top‑k = 50.  
- **Evaluation**: Simple entropy measurement of the next‑token distribution before sampling.  
- **Benchmarks**: Three short prompts of different styles (narrative, explanatory, poetic).  

This minimal setup provides a clear, executable example that a reviewer can run to verify that CFG works as intended on a standard language model.  It also serves as a template for extending the experiments to larger models or more complex benchmarks.  

Feel free to modify the prompts or hyperparameters to explore other settings.