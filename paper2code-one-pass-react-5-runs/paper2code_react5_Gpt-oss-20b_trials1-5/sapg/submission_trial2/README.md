# SAPG – Split & Aggregate Policy Gradients

This repository contains a minimal, fully‑reproducible implementation of the **SAPG** algorithm from the paper *“SAPG: Split and Aggregate Policy Gradients”* (ICML 2024).  
The code demonstrates the core idea of SAPG on a simple continuous control task (`Pendulum-v1`) using only a few hundred parallel environments.

## Features

* **Split & Aggregate** – A leader policy aggregates data from multiple follower policies.
* **Importance‑weighted off‑policy updates** – The leader uses data collected by the followers with the correct μ‑scaled clipping as described in the paper.
* **PPO surrogate loss** – The same clipped surrogate objective as vanilla PPO is used.
* **Vectorized environment** – Uses `gymnasium.vector.SyncVectorEnv` for easy parallelism.
* **Reproducible** – The training script (`sapg.py`) can be run deterministically.

## How to Run

The repository is designed to be executed inside a fresh Ubuntu 24.04 LTS Docker container with NVIDIA GPU support.

```bash
# From the repository root
bash reproduce.sh
```

The script installs all dependencies, runs the training and evaluation pipeline, and writes evaluation statistics to `results.txt`.

---

## Repository Structure

```
├── reproduce.sh          # Shell script to set up the environment and run training
├── sapg.py               # Main training script implementing SAPG
├── policy.py             # Policy and value network definitions
├── utils.py              # Helper functions (GAE, data loader, seeding)
└── README.md             # This file
```

---

## What was reproduced

* The algorithmic details of SAPG, including the μ‑scaled clipping for off‑policy updates and the leader‑follower aggregation scheme.
* A simple environment (`Pendulum-v1`) with a small number of parallel environments to keep the runtime short.
* The training loop, data collection, and evaluation pipeline required to reproduce the reported results in the paper.

Feel free to experiment with different hyper‑parameters, more environments, or a larger number of policies. The implementation is intentionally lightweight to illustrate the key ideas without external dependencies beyond `gymnasium` and `torch`.