# Simformer Toy Reproduction

This repository contains a minimal, fully reproducible implementation of a toy version of the **Simformer** model described in
> *All‑in‑one simulation‑based inference*  
> Manuel Gloeckler, Michael Deistler, Christian Weilbach, Frank Wood, Jakob H. Macke

The implementation focuses on a simple simulation‑based inference task:
- **Two‑Moons** dataset (2‑dimensional parameters, 2‑dimensional observations)
- Diffusion‑based score estimation using a lightweight Transformer‑style MLP
- Training via denoising score matching
- Posterior sampling via reverse diffusion

> **NOTE** – This is *not* the full implementation used in the paper.  
> It is a didactic, self‑contained toy that demonstrates the core ideas:
> 1. Joint modelling of parameters and data through a diffusion process.
> 2. Conditional inference by masking observed variables.
> 3. All‑conditional sampling (posterior, likelihood, etc.) from a single model.

## How to run

```bash
bash reproduce.sh
```

The script will:

1. Install the required Python packages (`jax`, `haiku`, `optax`, …) in the current environment.
2. Execute `scripts/train_and_eval.py`, which:
   * Generates a synthetic dataset of 20 000 `(θ, x)` pairs.
   * Trains a simple diffusion score model for 3 000 gradient steps.
   * Samples 5 000 posterior draws for a single synthetic observation.
   * Saves the posterior samples to `posterior_samples.csv`.

After the script finishes you will find:
- `posterior_samples.csv` – CSV with columns `theta_0, theta_1`.
- `training_log.txt` – Per‑step training loss (optional).
- `model_params.pkl` – Pickled model parameters (optional).

The entire pipeline can be re‑run on any machine with Python 3.10+ and a CUDA‑capable GPU
or a CPU‑only environment. The code is written to be **fully portable**:
no hard‑coded absolute paths are used.

## Expected Output

Inspecting the CSV:

```bash
head posterior_samples.csv
```

You should see something like:

```
theta_0,theta_1
0.12,-0.34
-0.01,0.78
...
```

These samples are drawn from the learned posterior distribution
`p(θ | x_obs)` for a synthetic observation `x_obs` that the script generates
on the fly.

## File Structure

```
├── README.md
├── reproduce.sh
├── requirements.txt
├── scripts/
│   └── train_and_eval.py
└── simformer/
    ├── __init__.py
    ├── model.py
    ├── trainer.py
    └── utils.py
```

- `simformer/` contains the core model (a lightweight Transformer‑style MLP),
  training utilities and helper functions.
- `scripts/train_and_eval.py` orchestrates data generation, training and sampling.
- The `reproduce.sh` script installs dependencies and runs the pipeline.

Feel free to tweak hyper‑parameters or replace the toy dataset with a more realistic simulator –
the structure is intentionally simple to keep the repository lightweight (< 1 GB) and fully reproducible.