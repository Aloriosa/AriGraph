# Batch-And-Match (BaM) Variational Inference

This repository contains a minimal, fully‑reproducible implementation of the *Batch and Match* (BaM) algorithm described in the paper *"Batch and match: black‑box variational inference with a score‑based divergence"*.

The code is written in pure Python using NumPy, SciPy and JAX for automatic differentiation.  
The reproduction script `reproduce.sh` installs the required packages and runs the experiment
on a synthetic Gaussian target. The script prints the final KL divergence between the target
and the variational approximation, the number of gradient evaluations, and the runtime.

## Repository structure

```
├── README.md
├── reproduce.sh
├── main.py
├── requirements.txt
└── .gitignore
```

## How to reproduce

```bash
# From the repository root
bash reproduce.sh
```

The script will:

1. Install the required Python packages (`jax`, `jaxlib`, `numpy`, `scipy`, `tqdm`).
2. Run `python main.py` with default arguments that reproduce the results reported
   in the toy experiments section of the paper (Gaussian target with dimension 16).
3. Output the following:
   * Final mean and covariance of the variational Gaussian.
   * KL divergence to the true Gaussian target.
   * Total number of gradient evaluations.
   * Runtime.

The whole process takes less than a minute on a modern laptop.

## Implementation notes

* **BaM** – Implements Algorithm 1 from the paper.  
  The updates for the mean and covariance are closed‑form and use the
  matrix square root from `scipy.linalg.sqrtm`.  
  A small batch size (`B=10`) is used to illustrate the stochastic nature.
* **Target** – A synthetic Gaussian target `p = N(mu_true, Sigma_true)` is generated
  with random mean and a positive‑definite covariance matrix.
* **Evaluation** – The KL divergence between two Gaussians can be computed analytically
  and is used as the final metric.
* **Dependencies** – Only lightweight packages are required.  No large datasets or
  heavy ML models are used, keeping the repository size far below 1 GB.

Feel free to modify the command line arguments of `main.py` to experiment with
different dimensions, batch sizes, regularisation strengths, etc.