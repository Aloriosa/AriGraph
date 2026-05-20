# Batch & Match (BaM) – Reproduction of *“Batch and match: black‑box variational inference with a score‑based divergence”*

This repository contains a minimal, self‑contained implementation of the **BaM** algorithm described in the paper
*Batch and match: black‑box variational inference with a score‑based divergence* (Cai et al., 2024).  
The code reproduces the main synthetic experiments on Gaussian target distributions (see Section 5.1 of the paper),
and it also implements a baseline ADVI method for comparison.

> **What is reproduced?**  
> • The BaM algorithm (Algorithm 1 in the paper).  
> • A simple ADVI baseline using stochastic ELBO optimisation.  
> • KL‑divergence curves for Gaussian targets of dimension 4, 16, 64, 256 (forward and reverse KL).  
> • A lightweight experimental script (`experiments/run_gaussian.py`) that outputs the results in CSV files.  

> **How to run?**  
> 1. Clone the repository.  
> 2. Run `bash reproduce.sh`.  
> 3. Results are written to `results/*.csv`; plots can be generated with any CSV‑viewer or by running the provided Jupyter notebook `experiments/plot_results.ipynb`.  

> **Environment**  
> The code uses JAX (with automatic GPU support) and Optax.  
> The `reproduce.sh` script installs the required packages with `pip`.  
> The repository size is < 10 MB (only source code, no large data or checkpoints).  

> **Author**  
> This reproduction was produced by an AI coding agent, following the paper and the instructions above.  
> It is meant to serve as a reference implementation for the community.