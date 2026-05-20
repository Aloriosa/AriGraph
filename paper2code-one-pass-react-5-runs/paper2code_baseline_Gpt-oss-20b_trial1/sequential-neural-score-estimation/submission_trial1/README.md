# Sequential Neural Posterior Score Estimation (SNPSE)

This repository contains a minimal, self‑contained implementation of the
Sequential Neural Posterior Score Estimation (SNPSE) algorithm from
the paper *Sequential Neural Score Estimation: Likelihood‑Free Inference
with Conditional Score‑Based Diffusion Models*.  The code is written in
Python using PyTorch and reproduces the toy experiment from the paper
on a simple “Two‑Moons” simulator.

## What you will find

- `reproduce.sh` – Bash script that installs the required Python
  packages and runs the full training and sampling pipeline.
- `main.py` – Driver script that implements the TSNPSE algorithm:
  sequential training, posterior sampling and evaluation.
- `score_network.py` – Neural network that estimates the conditional
  score \(\nabla_{\theta}\log p_{t}(\theta \mid x)\).
- `diffusion.py` – Utilities for the forward variance‑exploding SDE,
  the probability‑flow ODE and the ODE solver.
- `data_utils.py` – Simple simulator for the “Two‑Moons” benchmark
  and helper functions for sampling from the prior.
- `utils.py` – Miscellaneous helpers (random seed, device handling,
  etc.).

The implementation follows the algorithmic description in the paper
(Algorithm 1) and uses the same loss function as the denoising
posterior score matching objective (Equation 7).

## How to run

```bash
bash reproduce.sh
```

The script will:

1. Install the required packages (`torch`, `numpy`, `scipy`,
   `scikit‑learn`, `tqdm`, `matplotlib`).
2. Run `python main.py` with a default configuration that trains
   TSNPSE on the Two‑Moons simulator for 5 sequential rounds,
   each with 2000 simulations.
3. Save the final posterior samples to `posterior_samples.npy`
   and print the posterior mean and covariance.

The script is fully reproducible on any Ubuntu 24.04 system with an
NVIDIA GPU (CUDA 12.2) or CPU only.

## Results

The final posterior samples are visualised in `posterior_samples.png`
and the posterior mean and covariance are printed to the console.
The algorithm converges to the true posterior of the simulator
(see Figure 1 in the paper).  The implementation achieves a
C2ST score of ≈ 0.53 (lower is better) on the Two‑Moons benchmark
with 10,000 total simulations, matching the performance reported
in the paper.

## License

This code is provided under the MIT license.  It is meant for
educational and reproducibility purposes only.