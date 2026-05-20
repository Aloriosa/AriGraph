# Simformer‑Style Simulation‑Based Inference Reproduction

This repository contains a *minimal* implementation of the core ideas from the
“Simformer” paper: a transformer‑based score network trained on the joint
distribution of parameters and data and used with a diffusion model to
sample arbitrary conditionals (posterior, likelihood, etc.).

The goal of this reproduction is to provide a lightweight, fully‑self‑contained
example that can be executed in the evaluation environment (Ubuntu 24.04,
CPU‑only) and produces a small amount of output that can be inspected.

## What the code does

1. **Simulator** – a simple Gaussian‑linear simulator  
   \[
   \theta \sim \mathcal{N}(0,\;0.1\,I), \qquad
   x \mid \theta \sim \mathcal{N}(\theta,\;0.1\,I)
   \]
2. **Data generation** – 20 000 training pairs \((\theta,x)\).
3. **Simformer** – a transformer that takes a sequence of tokens
   (one per variable) and predicts the *score* \(\nabla \log p_t(x_t)\) for a
   noisy observation \(x_t\).  
   The tokenizer embeds the variable identity, the value and a binary
   condition flag.
4. **Training** – denoising score matching on the joint distribution
   with a simple VESDE noise schedule.
5. **Inference** – sampling the posterior \(p(\theta\mid x_o)\) via reverse
   diffusion.  The conditioning is enforced by keeping the observed data
   tokens fixed during the reverse walk.
6. **Output** –  
   * `posterior_samples.npy` – 10 000 posterior samples for a new
   synthetic observation.  
   * `posterior_summary.txt` – mean, standard deviation, and the true
   parameter used to generate the observation.

The code is intentionally simple (no external data files, no GPU
requirements) and is fully reproducible by running:

```bash
bash reproduce.sh
```

## Repository layout

```
├── README.md
├── reproduce.sh           # Driver script
├── requirements.txt
├── simformer.py           # Core Simformer implementation
├── data.py                # Simulator and dataset helpers
├── train.py               # Training script
├── inference.py           # Sampling script
└── utils.py               # Small utilities
```

All heavy artefacts (model checkpoints, generated data) are written
to disk by the scripts and are **not** committed to the repository.