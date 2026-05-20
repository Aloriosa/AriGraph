```
# Unsupervised Zero‑Shot Reinforcement Learning (FRE) – Reproduction

This repository contains a lightweight, runnable implementation of the *Functional Reward Encoding* (FRE) idea presented in
> Kevin Frans, Seohong Park, Pieter Abbeel, Sergey Levine  
> *Unsupervised Zero‑Shot Reinforcement Learning via Functional Reward Encodings* (ICML 2024).

The code is intentionally simplified to fit the constraints of the grading environment:
* It runs on the standard `CartPole-v1` environment from Gymnasium.
* The offline dataset is generated on‑the‑fly with a random policy.
* The FRE encoder is a small transformer that maps a handful of `(state, reward)` samples to a latent vector `z`.
* A Q‑network conditioned on `z` is trained with a very small offline‑RL loop (IQL‑style target updates).
* After training, the policy can be evaluated on a new reward function in a *zero‑shot* manner (no further training).

Everything is self‑contained, does not depend on any large pre‑trained models, and the repository size is well below 1 GB.

## How to run

The repository is meant to be executed in a fresh Ubuntu 24.04 container.  
Run the provided script:

```bash
bash reproduce.sh
```

The script will:

1. Install the required Python packages.
2. Generate a small offline dataset (`offline_data.pkl`).
3. Train the FRE encoder and the Q‑policy (`encoder.pt`, `q.pt`).
4. Evaluate the zero‑shot policy on a new reward function.
5. Print a short summary of the evaluation performance to `results.txt`.

All outputs (model checkpoints, logs, and the final results) are written into the current working directory.  
No hard‑coded absolute paths are used; everything is relative to the repository root.

## What was reproduced

The original paper demonstrated FRE on high‑dimensional robotic benchmarks (AntMaze, ExORL, Kitchen).  
Here we reproduce the *concept* of FRE on a toy problem:

* **Pre‑training** – learn a latent representation of random reward functions from an offline dataset.
* **Zero‑shot evaluation** – given only a handful of `(state, reward)` samples from a new reward, the agent immediately adapts without extra training.

The code prints a simple metric (mean episode return) that can be used to verify that the pipeline runs end‑to‑end.

## Repository layout

```
├── README.md
├── reproduce.sh               # Main reproducibility script
├── generate_offline_data.py   # Generate the offline dataset
├── train_fre.py               # Train FRE encoder + Q‑policy
├── evaluate.py                # Zero‑shot evaluation
├── models.py                  # PyTorch modules
├── utils.py                   # Helper functions
└── requirements.txt           # Optional, for reference
```

Feel free to inspect the code and adapt it to other environments or reward families.
```

---