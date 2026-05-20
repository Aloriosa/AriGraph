# Sequential Neural Score Estimation (SNPSE)

This repository contains a minimal, self‑contained implementation of the methods described in
> *Sequential Neural Score Estimation: Likelihood‑Free Inference with Conditional Score Based Diffusion Models*  
> (Sharrock, Simons, Liu & Beaumont, ICML 2024).

The implementation focuses on the **toy example** that is fully deterministic and does not require any heavy external libraries or large simulation budgets.  
It demonstrates the core idea of estimating the score of a posterior distribution using a conditional diffusion model and sampling from it via the probability‑flow ODE.

> **NOTE**  
> The full paper proposes a rich set of algorithms (NPSE, TSNPSE, SNPSE‑A/B/C) and extensive experiments on eight SBI benchmarks.  
> Re‑implementing all of those components would exceed the scope of a short exercise.  
> The code below is a *fully reproducible* toy demo that reproduces the key idea of the paper in a lightweight setting.

## How to run

```bash
bash reproduce.sh
```

The script will:
1. Install the required Python packages (NumPy, PyTorch, SciPy).
2. Run a short simulation of a toy Bayesian inference problem.
3. Train a conditional score network on synthetic data.
4. Sample from the approximate posterior using the probability‑flow ODE.
5. Write the generated samples to `samples.npy` and a small log file.

You can inspect the results in `samples.npy` (NumPy array of shape `[N, d]`) and `log.txt`.

## Repository contents

| File | Purpose |
|------|---------|
| `reproduce.sh` | Bash wrapper that installs dependencies and runs `python -m src.main`. |
| `requirements.txt` | Python package list. |
| `src/` | Python package containing the toy implementation. |
| `src/utils.py` | Helper functions for the diffusion process. |
| `src/main.py` | Entry point that orchestrates the toy experiment. |

Because this is a toy demo, the implementation uses:
- A 2‑dimensional Gaussian prior and likelihood.
- A variance‑exploding SDE for the forward process.
- A small MLP as the conditional score network.
- The probability‑flow ODE for sampling.

All code is written in pure Python and runs on CPU; you may use a GPU if you wish by setting the `CUDA_VISIBLE_DEVICES` environment variable.

---

Enjoy exploring the toy version of SNPSE! Feel free to extend it to the full benchmark experiments described in the paper.