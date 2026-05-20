# Batch and Match (BaM) Variational Inference

This repository contains a lightweight implementation of the **Batch and Match (BaM)** algorithm
introduced in the paper *“Batch and match: black-box variational inference with a score-based divergence”*.
It reproduces the core Gaussian target experiment described in Section 5 of the paper using only
`numpy`, `scipy`, and `argparse`.  The reproduction script installs the required Python
packages, runs the BaM algorithm on a synthetic Gaussian target, and prints the key
performance metrics (final variational parameters and KL divergences).

> **Note**  
> The implementation focuses on the main algorithmic ideas and does **not** reproduce all
> experiments from the paper (e.g. hierarchical models, deep generative models, or
> comparison with ADVI/GSM).  It is fully reproducible in the provided Docker
> environment and is well within the 1 GB size limit.

## Directory Structure

```
/home/submission/
├── README.md          # this file
├── reproduce.sh       # reproducibility script
├── baM.py             # BaM implementation and demo
└── utils.py           # small helper functions
```

## Reproducing the Results

1. **Build the container** (this step is done automatically by the judge).  
2. **Run the reproduction script**:

```bash
bash reproduce.sh
```

The script will:
1. Install `numpy` and `scipy`.
2. Execute `baM.py` with default parameters (10‑dimensional Gaussian target,
   200 iterations, batch size = 50, learning rate λ = 1.0).
3. Output the final variational mean and covariance, the analytic KL divergence
   between the target and the variational distribution, and an optional
   Monte‑Carlo estimate of the forward KL.

The results are printed to the console and saved to `results.txt`.

## Customizing the Experiment

You can modify the experiment parameters by passing command‑line arguments to
`baM.py`.  For example:

```bash
python3 baM.py --dim 20 --iterations 300 --batch 100 --lambda 0.5
```

Available arguments:

| Flag | Description |
|------|-------------|
| `--dim` | Dimensionality of the Gaussian target (default = 10) |
| `--iterations` | Number of BaM iterations (default = 200) |
| `--batch` | Batch size B (default = 50) |
| `--lambda` | Inverse regularization parameter λ (default = 1.0) |
| `--seed` | Random seed (default = 42) |

Feel free to experiment with different settings to observe the convergence
behaviour of BaM.

## License

This code is released under the MIT license.  See `LICENSE` for details.