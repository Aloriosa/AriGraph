# SAPG Reproduction (Toy Implementation)

This repository contains a minimal, self‑contained implementation of the **Split and Aggregate Policy Gradients (SAPG)** algorithm described in the paper *“SAPG: Split and Aggregate Policy Gradients”*.

> **What we reproduce**  
> • Two policies (a *leader* and a *follower*) trained on the `CartPole‑v1` environment.  
> • The leader learns from the follower’s data via an importance‑weighted off‑policy update that uses the clipping rule  
>   `clip(r, μ(1‑ε), μ(1+ε))` as described in the paper.  
> • The follower is trained only on its own on‑policy data.  
> • Both policies share a common backbone and have a small per‑policy embedding.  
> • Generalised Advantage Estimation (GAE), entropy regularisation, and a simple learning‑rate schedule are used.  
> • A vanilla PPO implementation is provided as a baseline.  
> • A single `reproduce.sh` script runs both experiments and writes the final episode returns to the `outputs/` directory.  

> **Why this is not a full reproduction**  
> The original paper evaluates SAPG on high‑dimensional, GPU‑accelerated dexterous manipulation tasks with tens of thousands of parallel environments.  Our toy implementation uses the discrete `CartPole‑v1` environment and a vectorised CPU backend.  It is meant to demonstrate the core algorithmic ideas rather than replicate the exact numbers reported in the paper.

## How to run

```bash
bash reproduce.sh
```

The script installs the required Python packages, trains both SAPG and vanilla PPO, and writes the episode returns to `outputs/results_sapg.txt` and `outputs/results_ppo.txt`.  The training logs are printed to the console.

---

If you want to experiment with the code:

* `train_sapg.py` – SAPG training loop.  
* `train_ppo.py` – Vanilla PPO baseline.  
* `utils.py` – Common network and helper functions.  

Feel free to tweak hyper‑parameters, change the number of environments, or extend the code to other Gym environments.