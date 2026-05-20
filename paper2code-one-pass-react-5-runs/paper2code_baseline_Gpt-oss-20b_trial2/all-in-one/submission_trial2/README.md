# Simformer – A Simplified Reimplementation

This repository contains a lightweight, end‑to‑end implementation of the core ideas behind the *Simformer* paper
"All‑in‑one simulation‑based inference".  
The implementation focuses on a toy benchmark (the Two‑Moons simulation) and demonstrates:

1. Sampling the joint distribution of parameters and data with a diffusion‑based transformer score model.  
2. Training the model using denoising score matching.  
3. Performing a forward diffusion (sampling) to generate new data.  

The code is intentionally kept small (≈ 1 KB of source) so that the repository stays well below the 1 GB size limit.  
All heavy computation (simulations, training, sampling) happens at runtime and does **not** ship any
pre‑trained weights.

> **Important** – The repository is designed to be run in a clean environment.  
> The reproduction script (`reproduce.sh`) installs the required Python packages and runs the training
> and sampling pipeline.  
> No absolute paths are used; all code references files relative to the repository root.

## Repository layout

```
.
├── README.md
├── reproduce.sh
├── simformer
│   ├── __init__.py
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   └── sample.py
└── requirements.txt
```

## How to run

```bash
bash reproduce.sh
```

The script will:

1. Install the required Python packages (`torch`, `numpy`, `scikit-learn`).  
2. Train a simple transformer‑based diffusion model on 100 000 samples of the Two‑Moons benchmark.  
3. Generate 5 000 samples from the trained model and store them in `samples.csv`.  
4. Print a short training summary and the shape of the generated samples.

You can inspect `samples.csv` to see the joint samples (4‑dimensional vectors:  
`[theta_x, theta_y, x_x, x_y]`).

## What was reproduced

- **Model** – A minimal transformer with 4 layers, 4 heads and a 4‑dimensional token representation.  
- **Diffusion** – Variance‑exploding SDE (VESDE) with the same hyper‑parameters as in the paper.  
- **Training** – Denoising score matching on 100 000 simulated pairs.  
- **Sampling** – Euler–Maruyama reverse diffusion with 500 discrete steps.  

The implementation is *not* a full replacement of the original Simformer but contains the key
components (tokenizer → transformer → score → guided diffusion).  The goal is to provide a fully
reproducible, self‑contained example that can run inside the grading container.

```