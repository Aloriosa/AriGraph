# Sequential Neural Posterior Score Estimation (SNPSE)

This repository contains a minimal but functional implementation of the
Sequential Neural Posterior Score Estimation (SNPSE) algorithm described in
the paper *Sequential Neural Score Estimation: Likelihood‑Free Inference
with Conditional Score Based Diffusion Models*.
The code reproduces the core method on a toy Gaussian‑linear
simulation‑based inference benchmark, trains the score network,
performs sequential inference with a truncated prior, and evaluates
the posterior approximation using a two‑sample classification test
(C2ST).  The implementation is fully self‑contained and can be
extended to the full suite of benchmarks in the paper with minor
modifications.

## Reproducibility

The repository is intentionally lightweight (no large artefacts).  
To reproduce the experiment run:

```bash
bash reproduce.sh
```

The script installs dependencies, trains the model and prints the
C2ST score for the Gaussian‑linear benchmark.

## Repository layout

```
/home/submission/
│
├── README.md
├── reproduce.sh
├── generated_reproduction.txt
├── src/
│   ├── __init__.py
│   ├── score_network.py
│   ├── simulator.py
│   ├── tsnpse.py
│   ├── utils.py
│   └── main.py
└── tests/          # optional unit tests (not part of the final repo)
```

## Key components

* **`score_network.py`** – MLP based conditional score network.
* **`simulator.py`** – Implements the Gaussian‑linear simulator and prior.
* **`tsnpse.py`** – Core TSNPSE training loop.
* **`utils.py`** – Helper functions for data handling, ODE solver,
  KDE for HPR estimation, and evaluation metrics.
* **`main.py`** – Driver script that runs the full training and
  evaluation pipeline.
* **`reproduce.sh`** – Bash script that installs dependencies and
  launches the experiment.

The implementation follows the algorithmic description in the paper:

1. **Forward diffusion (VE‑SDE)** – Adds Gaussian noise to the
   parameters.
2. **Conditional denoising score matching** – Learns a score
   network that estimates the gradient of the log‑posterior at each
   diffusion time.
3. **Sequential training** – After each round the highest‑probability
   region (HPR) of the current posterior is estimated with a KDE,
   and the next round samples are drawn from the prior truncated to
   this region.
4. **Posterior sampling** – Samples are obtained by integrating the
   probability‑flow ODE from the reference distribution back to the
   target posterior.
5. **Evaluation** – C2ST compares samples from the learned posterior
   with true posterior samples (which are analytically available for
   the Gaussian‑linear benchmark).

Feel free to extend the `simulator.py` module to other benchmark
simulators described in the paper – the rest of the pipeline is
generic.