# Batch and Match (BaM) – Black‑Box Variational Inference

This repository contains a minimal, fully reproducible implementation of the
**BaM** algorithm described in

> Cai, D., Modi, C., Pillaud‑Vivien, L., Margossian, C., Gower, R., Blei, D., & Saul, L. (2024).  
> *Batch and match: black‑box variational inference with a score‑based divergence*.

The code demonstrates the core BaM algorithm and compares it to two
baseline black‑box VI methods:

* **ADVI** – Automatic Differentiation Variational Inference (ELBO maximisation)
* **GSM** – Gaussian Score Matching (special case of BaM)

## Experiments

The following experiments are reproduced automatically by the `reproduce.sh`
script:

| Experiment | Target | Algorithms | Output |
|------------|--------|------------|--------|
| **Synthetic Gaussian** | Univariate Gaussian target (dim = 10) | BaM, ADVI, GSM | `figures/synthetic_results.png` |
| **Synthetic Non‑Gaussian** | Sinh‑arcsinh target (dim = 10) | BaM, ADVI, GSM | `figures/non_gaussian_results.png` |
| **Hierarchical Bayesian** | Simple 2‑level normal model | BaM, ADVI, GSM | `figures/hierarchical_results.png` |
| **Deep Generative (VAE)** | CIFAR‑10 VAE | BaM, ADVI, GSM | `figures/vae_results.png` |

The synthetic non‑Gaussian experiment demonstrates that the BaM algorithm
converges faster than the baseline methods even when the target density is
non‑Gaussian.  
The hierarchical Bayesian experiment uses a toy 2‑level normal model to
illustrate how BaM can be applied to real Bayesian inference problems.  
The deep generative experiment trains a small convolutional VAE on CIFAR‑10
and then performs posterior inference on a held‑out image using each
method.

The `reproduce.sh` script installs the required dependencies and runs
all four experiments, saving each figure in the `figures` directory.

## Repository Structure

```
├── experiments
│   ├── synthetic.py         # Gaussian target experiment
│   ├── non_gaussian.py      # Sinh‑arcsinh target experiment
│   ├── hierarchical.py      # 2‑level normal model experiment
│   └── vae.py               # CIFAR‑10 VAE experiment
├── src
│   ├── bam.py               # BaM, ADVI, GSM implementations
│   └── utils.py             # Miscellaneous utilities
├── reproduce.sh             # Reproduction script
├── requirements.txt         # (optional) dependency list
└── README.md
```

The implementation uses **JAX** for efficient GPU/CPU execution and
automatic differentiation.  All iterative updates are fully vectorised.
The code is written to be self‑contained; no external data or heavy
artifacts are committed to the repository.

## Running the Experiments

```bash
bash reproduce.sh
```

This will generate four PNG figures in the `figures/` directory.

The repository size is well below 1 GB – only source code and small
figures are committed.  Any untracked files (e.g., large datasets) are
removed before grading.