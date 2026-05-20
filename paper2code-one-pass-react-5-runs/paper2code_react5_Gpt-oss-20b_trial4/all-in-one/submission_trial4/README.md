# Simformer – All‑in‑One Simulation‑Based Inference

This repository contains a **minimal, reproducible implementation** of the
*Simformer* architecture described in the paper
*All‑in‑One Simulation‑Based Inference*.
The implementation focuses on the toy Two‑Moons simulator and demonstrates:

- Joint training of a transformer‑based score network on the joint distribution `p(θ, x)`.
- Random conditioning masks (joint, posterior, likelihood, random) during training.
- A generic tokenizer that works for arbitrary dimensional parameters and data.
- Guided reverse diffusion that allows sampling from arbitrary conditionals
  (posterior, likelihood, parameter marginals).
- Sampling of arbitrary conditionals beyond the posterior.
- A simple evaluation that reports the mean‑squared‑error between the
  true parameters and the inferred posterior samples.

The code is intentionally lightweight while still illustrating the core ideas
of the Simformer.  It can be used as a starting point for more elaborate
experiments or for extending the implementation to the full set of tasks
described in the paper.

## Reproduction

The reproducibility script installs the required packages (JAX with CUDA, Flax,
Optax, NumPy and tqdm) and then trains the Simformer on the Two‑Moons
simulator followed by a posterior sampling evaluation.

```bash
bash reproduce.sh
```

After the training and evaluation have finished, you will find the posterior
samples in `results/posterior_samples.npy`.  The script also prints a simple
mean‑squared‑error metric that quantifies how close the inferred posterior is
to the true parameters.

## Project structure

```
src/
├── __init__.py
├── attention_mask.py
├── evaluate.py
├── sde.py
├── simformer.py
├── simulator.py
├── tokenizer.py
├── train_simformer.py
└── utils.py
```

Each module is documented in the source code.  The key components are:

* `src/simulator.py` – toy Two‑Moons simulator.
* `src/tokenizer.py` – generic tokenizer mapping every variable to a token.
* `src/simformer.py` – transformer‑based score estimator.
* `src/sde.py` – forward and reverse diffusion for a simple VPSDE.
* `src/train_simformer.py` – training loop with random conditioning masks.
* `src/evaluate.py` – guided reverse diffusion to sample from arbitrary
  conditionals, here used for posterior sampling.

Feel free to modify the hyper‑parameters in `src/train_simformer.py`
or `src/evaluate.py` to experiment with different settings.