# Batch and Match (BaM) – Reproduction of the 2024 ICML paper

This repository contains a lightweight implementation of the **Batch and Match (BaM)** algorithm for black‑box variational inference with a score‑based divergence.  
The code reproduces the toy experiments reported in the paper:

* a Gaussian target distribution with varying dimensions,
* convergence diagnostics (forward and reverse KL divergence),
* comparison against a simple stochastic ELBO optimiser (ADVI) and the Gaussian Score Matching (GSM) algorithm.

The repository is intentionally small (≈ 200 KB of source code) and does **not** ship any large data or pre‑trained models. The only external dependency is `numpy`/`scipy` (≈ 1 MB).  

### How to run

```bash
# From the repository root:
bash reproduce.sh
```

The `reproduce.sh` script will:

1. Install the required Python packages (`numpy`, `scipy`).
2. Run the toy Gaussian experiment (`experiments/run_gaussian.py`).
3. Print the KL diagnostics and a convergence plot.

The script outputs a PNG file `results/kl_convergence.png` in the repository root.

> **Note**: The code is written to run on CPU, but will automatically use a GPU if `jax` is available and a CUDA device is detected.  
> If you want to force CPU usage, set `export FORCE_CPU=1`.

### Repository layout

```
src/          – core BaM implementation and utilities.
experiments/  – scripts that set up toy experiments.
reproduce.sh  – the reproducibility driver.
README.md     – this file.
```

### Expected Output

After running `reproduce.sh`, you should see output similar to:

```
Iteration 0  KL(p||q)=0.87  KL(q||p)=0.87  KL_kl(q||p)=0.87
Iteration 1  KL(p||q)=0.61  KL(q||p)=0.61  KL_kl(q||p)=0.61
...
Iteration 19 KL(p||q)=0.01  KL(q||p)=0.01  KL_kl(q||p)=0.01