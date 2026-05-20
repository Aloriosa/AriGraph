#!/usr/bin/env bash
# Reproduce the synthetic, non‑Gaussian, hierarchical, and VAE experiments.

set -euo pipefail

# 1. Install required packages
python3 -m pip install --upgrade pip
python3 -m pip install -q \
    jax==0.4.23 \
    jaxlib==0.4.23 \
    optax==0.1.6 \
    matplotlib==3.8.4 \
    numpy==1.26.4 \
    tensorflow-datasets==4.9.0

# 2. Run experiments
python3 - <<'PYCODE'
from experiments.synthetic import run_experiment
from experiments.non_gaussian import run_non_gaussian
from experiments.hierarchical import run_hierarchical
from experiments.vae import run_vae_experiment

print("\nRunning synthetic Gaussian experiment…")
run_experiment(D=10, T=200, B=20, seed=42)

print("\nRunning non‑Gaussian sinh‑arcsinh experiment…")
run_non_gaussian(D=10, T=200, B=20, seed=42)

print("\nRunning hierarchical Bayesian experiment…")
run_hierarchical(T=200, B=20, seed=42)

print("\nRunning VAE posterior inference experiment…")
run_vae_experiment(seed=42)

print("\nAll experiments completed. Figures are in the 'figures/' directory.")
PYCODE