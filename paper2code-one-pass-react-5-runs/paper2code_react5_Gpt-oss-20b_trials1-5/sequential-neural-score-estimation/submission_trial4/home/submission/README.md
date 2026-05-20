# Sequential Neural Posterior Score Estimation (SNPSE)

This repository contains a minimal, fully reproducible implementation of the **Sequential Neural Posterior Score Estimation (SNPSE)** method described in  
> Louis Sharrock, Jack Simons, Song Liu, Mark Beaumont – *Sequential Neural Score Estimation: Likelihood‑Free Inference with Conditional Score Based Diffusion Models*.

The implementation focuses on the **Truncated Sequential NPSE (TSNPSE)** variant for the Gaussian‑linear benchmark, but the code is generic enough to be extended to other simulator‑based inference problems.

## Reproduction

On a fresh machine, run

```bash
bash reproduce.sh
```

The script will:

1. Install the required Python packages.
2. Train the TSNPSE model for the Gaussian‑linear benchmark.
3. Save the trained model, posterior samples, and a short report in the `outputs/` directory.

The entire process takes a few minutes on a modern CPU and a few seconds on a single NVIDIA GPU (the script automatically uses CUDA if available).

## Repository Structure

```
/home/submission/
├── reproduce.sh          # Main entry point
├── README.md
├── requirements.txt
├── config.yaml           # Hyper‑parameter configuration
├── outputs/              # Generated results
│   ├── samples.npy
│   ├── posterior_mean.txt
│   └── report.txt
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py
│   ├── model.py
│   ├── training.py
│   ├── utils.py
│   ├── sampling.py
│   └── main.py
└── experiments/          # (empty, placeholder for future extensions)
```

## What was reproduced

The script reproduces the core idea of TSNPSE:

- **Training**: The model learns the score of the perturbed posterior `∇_θ log p_t(θ_t | x_obs)` through the *conditional denoising posterior score matching* objective, which includes both the diffusion score and the likelihood gradient.
- **Sequential update**: After each round, the posterior approximation is used to construct a truncated prior, focusing subsequent simulations on the high‑probability region of the target posterior.
- **Sampling**: Posterior samples are drawn by integrating the probability‑flow ODE from the reference distribution back to the data space.
- **Evaluation**: For the Gaussian‑linear benchmark the posterior mean and variance are compared to their analytic values. A simple report with the squared error and variance error is written to `outputs/report.txt`.

This demonstrates that the implementation can recover the correct posterior distribution with a small simulation budget.

## Extending to other benchmarks

The code is written generically. To experiment on a different simulator:

1. **Define a new simulator** in `src/dataset.py` (e.g., the two‑moons benchmark).  
2. **Adjust the prior** and observation `x_obs` accordingly.  
3. **(Optional)** Update `config.yaml` to change hyper‑parameters such as the number of rounds, batch size, or learning rate.

The training pipeline will automatically handle the sequential loop, dataset construction, and sampling.