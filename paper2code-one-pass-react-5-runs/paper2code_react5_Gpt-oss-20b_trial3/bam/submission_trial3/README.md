# BaM Reproduction Repository

This repository contains a minimal, self‑contained implementation of the **Batch and Match (BaM)** algorithm described in
> *Batch and match: black‑box variational inference with a score‑based divergence*  
> (Cai et al., 2024).

The code reproduces the synthetic Gaussian experiments from the paper, compares BaM against two baselines
(ADVI and GSM), and records the evidence lower bound (ELBO), forward & reverse KL‑divergences and the
score‑based divergence for each iteration.

All computations are performed in **JAX** so that the code runs on CPU or GPU without any modifications.

## Reproducibility

To reproduce the results:

```bash
bash reproduce.sh
```

The script installs the required dependencies (`jax`, `jaxlib`, `optax`, `tqdm`, `numpy`) and runs
`main.py`.  The results are written to a file named `results.csv` in the current directory.

The `results.csv` file contains the following columns:

| method | iter | kl_forward | kl_reverse | score_div |
|--------|------|------------|------------|-----------|
| BaM    | 0‑199 | …          | …          | …         |
| ADVI   | 200  | …          | …          | …         |
| GSM    | 0‑199 | …          | …          | …         |

The script uses a random seed of `42`, dimensionality `16`, batch size `256`, regularisation
parameter `λ = 1.0`, and 200 iterations for each algorithm.

## Implementation Notes

* **BaM** – Implements the batch‑and‑match update with the closed‑form solution of the quadratic matrix equation.
* **ADVI** – Uses Adam optimizer over the reparameterised ELBO.
* **GSM** – Implements the Gaussian Score‑Matching updates from Modi et al. (2023) with per‑sample updates averaged over a batch.
* All algorithms use a multivariate Gaussian variational family with full covariance.
* The score‑based divergence is evaluated empirically on a fresh batch of 100 samples at each iteration.

## License

MIT license.  No heavy artifacts are committed – only source code and a short README.