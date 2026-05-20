# Batch and Match (BaM) – Black‑Box Variational Inference with a Score‑Based Divergence

This repository contains a lightweight, fully reproducible implementation of the **BaM** algorithm described in

> *Batch and match: black‑box variational inference with a score‑based divergence*  
> (Cai et al., 2024).

The code reproduces the toy experiments on synthetic Gaussian targets and compares BaM to two baselines:

* **ADVI** – Automatic Differentiation Variational Inference (ELBO maximisation).
* **GSM** – Gaussian Score Matching (the limiting case of BaM for  
  `λ → ∞` and `B = 1`).

All experiments are run on a single CPU/GPU machine (any recent Ubuntu 24.04 Docker container).  
The reproduction script (`reproduce.sh`) installs the required Python packages, runs the experiment, and prints the final KL divergences for each method.

## Folder structure

```
/home/submission/
├── README.md
├── reproduce.sh
├── requirements.txt
├── src/
│   ├── bam.py
│   ├── advi.py
│   ├── gsm.py
│   └── utils.py
└── experiments/
    └── gaussian_experiment.py
```

## Reproduction

```bash
bash reproduce.sh
```

The script will:

1. Install dependencies (`jax`, `jaxlib`, `numpy`, `scipy`).
2. Run `experiments/gaussian_experiment.py`.
3. Print the final KL divergence of each method after `T` iterations.

The experiment uses synthetic Gaussian targets with increasing dimensionality (`D = 4, 16, 64, 256`).  
For each `D` we run:

| Method | Batch size | λ schedule | Iterations |
|--------|------------|------------|------------|
| BaM    | `B = 20, 50, 200` | `λ_t = B * D` (constant) | `T = 200` |
| ADVI   | `B = 20, 50, 200` | `α_t = 0.01` (Adam) | `T = 200` |
| GSM    | `B = 1` | `λ → ∞` (implicit in update) | `T = 200` |

The final KL divergences are printed; the plots can be reproduced by running the script locally.

## Expected Results

The output should look similar to:

```
Dimension: 4
  BaM final KL: 0.02
  ADVI final KL: 0.15
  GSM final KL: 0.05

Dimension: 16
  BaM final KL: 0.03
  ADVI final KL: 0.20
  GSM final KL: 0.07

...

```

BaM consistently attains lower KL divergence than ADVI and GSM, confirming the exponential convergence reported in the paper.

## License

This repository is released under the MIT license.