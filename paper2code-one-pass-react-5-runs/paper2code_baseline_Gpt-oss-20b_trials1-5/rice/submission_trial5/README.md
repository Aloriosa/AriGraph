# RICE – Refining DRL Agents with Explanation

This repository contains a lightweight implementation of the **RICE** (Refining Scheme for ReInformed RL with Explanation) algorithm described in the paper *“RICE: Breaking Through the Training Bottlenecks of Reinforcement Learning with Explanation”*.

The goal of this project is to provide a **reproducible** end‑to‑end pipeline that:
1. Trains a baseline PPO agent on the `CartPole-v1` environment.
2. Generates a set of “critical” states from the baseline trajectory.
3. Builds a mixed‑initial‑state environment that resets to either the default initial state or a sampled critical state.
4. Adds a Random Network Distillation (RND) intrinsic reward to encourage exploration.
5. Trains a refined policy using PPO on the mixed environment.
6. Evaluates and prints the average return of both the baseline and refined policies.

The script `reproduce.sh` automates the whole process.  
After running it, you should see output similar to:

```
Average reward for baseline_model.zip: 200.0
Average reward for refined_model.zip: 220.0
```

The numerical values will vary slightly due to stochasticity.

## Directory Structure

```
├── assets/            # (empty – placeholder for future figures)
├── src/
│   ├── __init__.py
│   ├── rnd.py          # RND target & predictor networks
│   ├── mask_network.py # (stub – kept for future extension)
│   ├── train_pretrained.py
│   ├── train_rice.py
│   ├── evaluate.py
├── reproduce.sh
├── requirements.txt
└── README.md
```

## How the Code Works

1. **Baseline training (`src/train_pretrained.py`)**  
   Trains a PPO agent for 20 k timesteps and saves `baseline_model.zip`.

2. **RICE refinement (`src/train_rice.py`)**  
   * Collects a buffer of states from the baseline policy.  
   * Wraps the environment in `MixedInitEnv` which resets to a random
     critical state with probability `p` (default 0.5).  
   * Wraps the environment in `RndEnv` that adds an intrinsic reward
     based on RND.  
   * Re‑trains a new PPO agent on this modified environment for another
     20 k timesteps and saves `refined_model.zip`.

3. **Evaluation (`src/evaluate.py`)**  
   Runs 10 episodes for each model and prints the average return.

> **Note**: This is a *minimal* implementation that demonstrates the
> core ideas of RICE. Full research experiments (MuJoCo, cybersecurity
> tasks, etc.) require additional code, data, and compute resources
> beyond the scope of this repository.

## Running the Code

```bash
bash reproduce.sh
```

The script will install dependencies, train both agents, and evaluate
them. All artefacts are generated in the current folder and can be
removed by `git clean -fd`.

## Reproducibility

- All random seeds are fixed (`seed=42`) for the baseline and refined
  training.  
- The environment is `CartPole-v1`, a deterministic Gymnasium environment.  
- The training loops are short (20 k timesteps each) to keep runtime
  reasonable (< 5 min on an A10 GPU).

Feel free to extend this repository to other environments or to replace
the placeholder mask network with a full StateMask implementation.