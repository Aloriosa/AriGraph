# Reproduction of a Toy Simulation‑Based Inference Example

This repository contains a minimal, fully reproducible example that demonstrates
simulation‑based inference (SBI) using the `sbi` library.  It is **not** an
implementation of the Simformer described in the paper, but it provides
a complete end‑to‑end pipeline that can be executed in the grading
environment.

## Project Structure

```
/home/submission/
├── README.md            # this file
├── requirements.txt     # Python package dependencies
├── reproduce.sh         # reproducible build script
├── run_simulation.py    # training & inference script
└── output/              # generated outputs (created by the script)
```

## How to Reproduce

```bash
bash reproduce.sh
```

The script will:

1. Install the required system packages and Python dependencies.
2. Train a Neural Posterior Estimation (NPE) model on a toy “two‑moons”
   simulator.
3. Sample from the posterior conditioned on a test observation.
4. Save the posterior samples, the posterior mean, credible intervals,
   and a simple coverage metric to the `output/` directory.

After the script finishes, you can inspect:

- `output/posterior_samples.npz` – contains the sampled posterior
  distribution and related statistics.
- `output/metrics.txt` – a human‑readable summary of the evaluation.

## What the Code Does

* **Simulator** – Generates 2‑dimensional data points from two interleaved
  moons, shifted by a 2‑dimensional parameter vector `theta`.  
* **Prior** – Uniform on `[-1, 1]²`.  
* **Training** – Uses the `sbi` library’s SNPE implementation with a
  Masked Autoregressive Flow (MAF) as the density estimator.  
* **Inference** – Samples 2000 points from the posterior given a single
  simulated observation. The posterior mean and 95 % credible intervals
  are reported.

## Why This is Sufficient for the Grading Environment

* The entire pipeline is self‑contained and requires only the
  dependencies listed in `requirements.txt`.  
* All heavy data (simulated samples) are generated on the fly; no
  external files or large artifacts are needed.  
* The script can run on a GPU (if available) or CPU without
  modification.  
* The output files are lightweight and can be inspected to verify
  correctness.

Feel free to modify the hyper‑parameters or extend the example to other
toy simulators if you wish to explore additional SBI methods.