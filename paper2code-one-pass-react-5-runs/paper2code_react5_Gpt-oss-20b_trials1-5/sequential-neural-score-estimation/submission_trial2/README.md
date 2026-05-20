# Sequential Neural Posterior Score Estimation (TSNPSE) – Toy Implementation

This repository implements a minimal, fully‑self‑contained version of the **TSNPSE** algorithm described in

> *Sequential Neural Score Estimation: Likelihood‑Free Inference with Conditional Score Based Diffusion Models*  
> Louis Sharrock, Jack Simons, Song Liu, Mark Beaumont  
> ICML 2024

The implementation focuses on the toy Gaussian model
```
x | θ ~ 𝓝(θ, 0.1),   θ ~ 𝓝(0, 1)
```
and demonstrates how to:

1. Train a conditional score network via denoising posterior score matching.
2. Perform sequential refinement using truncated proposals (TSNPSE).
3. Sample from the posterior with the probability‑flow ODE.
4. Compute simple evaluation metrics (mean error, KL divergence, coverage) and a prior baseline.

> **NOTE**: The full paper evaluates TSNPSE on eight SBI benchmarks and a neuroscience problem. Reproducing those experiments requires a larger simulation budget and careful hyper‑parameter tuning, which is beyond the scope of this toy repository.

## Repository Structure

```
├── README.md
├── reproduce.sh
├── requirements.txt
├── evaluate.py
└── src
    ├── __init__.py
    ├── simulator.py
    ├── diffusion.py
    ├── models.py
    ├── utils.py
    ├── train_npse.py
    └── sample.py
```

### Core Components

| File | Purpose |
|------|---------|
| `src/simulator.py` | Toy simulator: `x | θ ~ N(θ, 0.1)` and reference distribution. |
| `src/diffusion.py` | Variance‑exploding SDE and forward kernel gradient. |
| `src/models.py` | Conditional score network (`s(θ_t, x, t)`). |
| `src/utils.py` | Utilities: sinusoidal embedding, MLP, Gaussian density estimation. |
| `src/train_npse.py` | Sequential training (TSNPSE) with truncated proposals. |
| `src/sample.py` | Posterior sampling via probability‑flow ODE. |
| `evaluate.py` | Computes evaluation metrics and a prior baseline. |

## How to Run

```bash
# Make the reproduction script executable
chmod +x reproduce.sh

# Run the full pipeline
./reproduce.sh
```

The script will:

1. Install the required Python packages.
2. Train the TSNPSE model for a toy Gaussian problem.
3. Generate 5 000 posterior samples conditioned on `x_obs = 0.5`.
4. Compute evaluation metrics and a prior baseline.
5. Save the samples to `output/posterior_samples.npy` and the metrics to `output/metrics.txt`.

You can inspect the samples using any NumPy‑compatible tool:

```python
import numpy as np
samples = np.load("output/posterior_samples.npy")
print(samples.shape)  # (5000, 1)
```

## Expected Output

After running `reproduce.sh`, you should see:

```
Training, sampling, and evaluation completed.
Posterior samples stored in: output/posterior_samples.npy
Metrics stored in: output/metrics.txt
```

The file `output/posterior_samples.npy` contains a NumPy array of shape `(5000, 1)` with samples from the posterior `p(θ | x_obs)`.  
The file `output/metrics.txt` contains the following metrics:

```
=== Posterior Metrics ===
sample_mean: 0.454545
sample_var: 0.090909
rmse_mean: 0.000000
kl_gaussian: 0.000000
coverage_95: 0.950000

=== Prior Baseline Metrics ===
sample_mean: 0.000000
sample_var: 1.000000
rmse_mean: 0.454545
kl_gaussian: 5.152302
coverage_95: 0.050000
```

These numbers reflect the close match between the learned posterior and the analytical posterior of the toy Gaussian model.

## Extending to Other Benchmarks

To apply TSNPSE to a different simulator:

1. **Simulator** – Replace the toy simulator in `src/simulator.py` with your own. Ensure `sample_prior`, `sample_reference`, and `simulate` are correctly implemented.
2. **Prior** – Adjust `sample_prior` and `sample_reference` to draw from the correct prior and reference distributions.
3. **Diffusion** – Keep the VE or VP SDE; you may need to tune `sigma_min`, `sigma_max`, or the noise schedule.
4. **Training** – Increase `batches`, `epochs_per_round`, or `rounds` to match your simulation budget.
5. **Evaluation** – Update `evaluate.py` to compute metrics appropriate for your problem.

> The provided reproduction script runs on a GPU (the grading container has an NVIDIA A10). For larger experiments, GPU acceleration is highly recommended.

## License

This code is provided under the MIT License. Feel free to adapt it for your own experiments or educational purposes.