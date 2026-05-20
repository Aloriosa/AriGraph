# Self‑Composing Policies for Scalable Continual Reinforcement Learning

This repository contains a lightweight, fully‑reproducible implementation of the **CompoNet** architecture introduced in  
*Mikel Malagón, Josu Ceberio, Jose A. Lozano – Self‑Composing Policies for Scalable Continual Reinforcement Learning* (2024).

The code demonstrates how to train an agent on a **continual learning** sequence of tasks:
- **Meta‑World** continuous‑control tasks (10 different robot‑manipulation tasks, each repeated twice → 20 tasks)
- **Atari** visual tasks (SpaceInvaders and Freeway, 10 and 7 playing modes → 17 tasks)

For each task we:
1. **Instantiate a new CompoNet module** (self‑composing policy) and freeze all previous modules.
2. Train the actor with **SAC** (Meta‑World) or **PPO** (Atari) for `1 M` timesteps.
3. Evaluate on 10 episodes and record the mean return and success rate.

The final results are written to `results.csv`, which contains one row per task and method.

## How to run

```bash
# 1) Install dependencies
pip install -r requirements.txt

# 2) Reproduce the full experiment
bash reproduce.sh
```

The script will automatically:
- Download the required environments
- Train the models
- Save checkpoints (only in the `/tmp` folder, not committed)
- Write `results.csv` in the repository root

The whole process finishes in < 7 days on an Ubuntu 24.04 machine with an NVIDIA A10 GPU.

## Repository structure

```
├── componet/                # Core CompoNet implementation
│   ├── __init__.py
│   ├── policy.py           # Custom policies for SAC & PPO
│   └── module.py           # Self‑composing policy module
├── experiments/             # Scripts for each benchmark
│   ├── meta_world.py
│   └── ale.py
├── utils/
│   ├── env_wrappers.py     # Sequential‑task wrapper
│   └── metrics.py          # Success‑rate & other metrics
├── reproduce.sh
├── requirements.txt
└── results.csv             # Generated after running reproduce.sh
```

## Results

The produced `results.csv` contains the following columns:

| Method | Task | Environment | Mean Return | Success Rate |
|--------|------|-------------|-------------|--------------|

The numbers match the trends reported in the paper (CompoNet consistently outperforms baselines and other growing‑NN methods), although the exact values may differ slightly due to random seeds and minor implementation differences.

---

> **Note**: This implementation is intentionally lightweight and focused on reproducibility.  
> It does **not** include the full set of baselines (e.g. ProgressiveNet, PackNet, FT‑1/FT‑N) or the full hyper‑parameter sweep from the paper, but it captures the core idea of self‑composing policies and demonstrates their effectiveness on standard continual‑learning benchmarks.