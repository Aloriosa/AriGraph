# TSNPSE – Reproduction of Sequential Neural Posterior Score Estimation

This repository contains a minimal but faithful implementation of the **TSNPSE** algorithm
presented in the paper

> *Sequential Neural Posterior Score Estimation: Likelihood‑Free Inference with Conditional Score Based Diffusion Models*  
> Louis Sharrock, Jack Simons, Song Liu & Mark Beaumont (ICML 2024).

The code demonstrates the following key ideas:

1. **Simulation‑based inference** – synthetic data are generated from a forward simulator
   (the “Two‑Moons” benchmark used in the paper).
2. **Conditional diffusion model** – a small MLP learns the posterior score
   `∇θ log p_t(θ_t | x)`.
3. **Sequential refinement** – the prior is iteratively truncated to the
   high‑probability region of the current posterior estimate.
4. **Deterministic sampling** – the probability‑flow ODE is integrated to draw
   samples from the learned posterior.
5. **Evaluation** – posterior predictive mean/variance are printed and a simple
   C2ST (classification‑based two‑sample test) between the approximate posterior
   and the ground‑truth posterior (estimated by a short MCMC run) is reported.

The implementation is intentionally lightweight so that it can be run on a
fresh Ubuntu 24.04 container with a single GPU in less than 7 days.

> **Important**: The repository does **not** ship any heavy artefacts (simulation
> outputs, checkpoints, etc.) – only source code and the tiny result files
> produced by the reproduction script.

## Usage

```bash
# From the repository root
bash reproduce.sh
```

The script will install the required dependencies, run the experiment and
write the results to the `results/` directory.  
The most important outputs are:

* `results/posterior_samples.npy` – posterior samples from the final TSNPSE round.
* `results/summary.txt` – posterior mean, variance, predictive mean/variance and
  a C2ST score.

## Code structure

```
├── reproduce.sh              # installation & execution
├── README.md
└── src
    ├── main.py               # driver script
    ├── simulator.py          # Two‑Moons simulator & prior sampler
    ├── model.py              # Conditional score MLP
    └── utils.py              # Small helpers
```

The implementation follows the paper closely:

* The **VE SDE** forward noising process is used.
* The **probability‑flow ODE** drift is `-0.5 σ_t² ⋅ score`, i.e. the correct
  sign as derived in the paper.
* Training uses the **conditional denoising posterior score‑matching loss**
  (essentially MSE between predicted and true score).
* The **truncated prior** is approximated by the posterior samples obtained in
  the previous round (this is a pragmatic simplification of the HPR
  computation described in the paper).
* A simple **C2ST** classifier is trained to distinguish samples from the
  approximate posterior and the ground‑truth posterior obtained by a short
  MCMC run (Pyro).

Feel free to tweak the hyper‑parameters (`N_SIMULATIONS`, `N_ROUNDS`,
`BATCH_SIZE`, etc.) to explore the trade‑off between simulation cost and
posterior quality.