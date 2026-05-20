# Sequential Neural Posterior Score Estimation (TSNPSE)

This repository contains a lightweight implementation of the **TSNPSE** algorithm from the paper  
> L. Sharrock, J. Simons, S. Liu, & M. Beaumont,  
> *Sequential Neural Score Estimation: Likelihood‑Free Inference with Conditional Score Based Diffusion Models* (ICML 2024).

The code implements both the truncated‑proposal variant (TSNPSE) and a simple SNPE baseline for comparison.  It runs on a small set of toy benchmark simulators:

- Gaussian Linear
- Gaussian Mixture
- Two‑Moons

The pipeline trains a conditional score network sequentially, truncates the proposal prior, and samples from the learned posterior using the probability‑flow ODE.  A baseline SNPE model is also trained on the same data and its posterior samples are saved for easy comparison.

## Features

- Conditional score network with sinusoidal time embedding
- Variance‑exploding SDE (VE‑SDE)
- Truncated‑prior sampling via KDE and ε‑threshold
- Probability‑flow ODE sampling
- Sequential training loop with data accumulation
- Diagnostic metrics: posterior mean, covariance, KL divergence (Gaussian‑linear), coverage, ESS
- Simple SNPE baseline (diagonal Gaussian posterior)

## Installation

```bash
# Make the reproduce script executable
chmod +x reproduce.sh

# Install Python dependencies
./reproduce.sh
```

The script will install the required packages (`torch>=2.0`, `tqdm`, `numpy`, `scikit-learn`, `scipy`) and run the experiments.

## Running the experiments

```bash
python3 main.py \
    --output-dir ./output \
    --rounds 5 \
    --sims-per-round 5000 \
    --epochs 15 \
    --baseline-epochs 15
```

This will run the sequential TSNPSE algorithm for 5 rounds with 5 000 simulations per round and a baseline SNPE model trained for the same number of epochs.

## Output

For each benchmark, the following files are written:

| File | Content |
|------|---------|
| `output/<benchmark>/tsnpse_samples.npy` | 5 000 posterior draws produced by TSNPSE (`float32`, shape `(5000, dim)` ) |
| `output/<benchmark>/snpe_samples.npy` | 5 000 posterior draws produced by the SNPE baseline |
| `output/<benchmark>/metrics.json` | JSON with diagnostics (means, covariances, KL, coverage, ESS, etc.) |

You can load the samples in Python and inspect the posterior:

```python
import numpy as np
import matplotlib.pyplot as plt

samples = np.load('output/gaussian_linear/tsnpse_samples.npy')
plt.scatter(samples[:,0], samples[:,1], alpha=0.5)
plt.title('Posterior samples – Gaussian Linear (TSNPSE)')
plt.show()
```

## Extending to new simulators

1. **Add a simulator**  
   Implement a function `simulate(batch_size, device, theta0=None)` that returns `(theta, x)` for the desired benchmark.  

2. **Add a benchmark entry**  
   In `BENCHMARKS`, add a dictionary with:
   * `name`: string identifier  
   * `dim_theta`: dimension of parameters  
   * `dim_x`: dimension of observations  
   * `sim_func`: the simulator function defined in step 1  
   * `prior_sampler`: a function that draws samples from the prior  
   * `x_obs`: observed data (torch tensor)  

3. **(Optional) Add analytic posterior**  
   If you know the analytic posterior, provide a helper that returns the mean and covariance to populate `ground_truth` in `metrics.json`.  

## Contact

For questions or bug reports, please open an issue on the GitHub repository.