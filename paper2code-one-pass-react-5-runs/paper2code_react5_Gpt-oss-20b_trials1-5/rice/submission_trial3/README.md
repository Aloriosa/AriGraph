# RICE – Refining Reinforcement Learning with Explanation

This repository contains a **minimal, self‑contained** implementation of the RICE framework described in the paper *“RICE: Breaking Through the Training Bottlenecks of Reinforcement Learning with Explanation”*.  
The goal is to provide code that:

1. **Pre‑trains** a PPO policy on a simple Gymnasium environment (CartPole-v1 by default).  
2. **Trains a mask network** that learns which time‑steps are most critical for the agent’s performance.  
3. **Refines** the policy by resetting the environment to a mixture of the default initial state and the identified critical states, while adding an exploration bonus from Random Network Distillation (RND).  
4. **Evaluates** the final policy and reports the average return.

> **NOTE**: This is a **toy implementation** that demonstrates the core ideas of RICE. It is **not** a full replication of the experiments in the paper (which involve MuJoCo, cryptocurrency mining, cyber‑defense, autonomous driving, and malware mutation). The implementation focuses on correctness, reproducibility, and easy extension to other environments.

## Directory Structure

```
├── requirements.txt          # Python dependencies
├── scripts/
│   └── reproduce.sh          # Full reproduction script
├── src/
│   ├── env_utils.py          # Helper to set environment state
│   ├── mask_network.py       # Mask network (step‑importance predictor)
│   ├── rnd.py                # RND modules
│   └── rice.py               # RICE algorithm
└── README.md
```

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full reproduction script
bash scripts/reproduce.sh
```

The script will:

1. Train a PPO policy for 10,000 timesteps (pre‑training).  
2. Train the mask network for 3 epochs.  
3. Refine the policy for 50 iterations (each collects ~200 steps).  
4. Evaluate the refined policy over 20 episodes and print the average reward.

All output (logs, intermediate models) are stored in the repository root.

## Extending to Other Environments

1. **Change the environment**: Edit `env_name` in `scripts/reproduce.sh` and `scripts/reproduce.sh` (the Python block that creates the pre‑trained policy).  
2. **Adjust hyper‑parameters**: `RICE.__init__` accepts `p`, `lam`, `alpha`, etc.  
3. **Handle state resetting**: The current `set_env_state` works for environments that expose a `state` attribute (e.g., CartPole, MountainCar). For custom environments, extend `env_utils.set_env_state` accordingly.

## Code Overview

- **`src/mask_network.py`** – A lightweight MLP that outputs a mask probability for each state.  
- **`src/rnd.py`** – Implements the target and predictor networks used in Random Network Distillation.  
- **`src/env_utils.py`** – Helper to directly set the internal state of an environment.  
- **`src/rice.py`** – Orchestrates the entire pipeline: pre‑training, mask training, refinement, and evaluation.  
- **`scripts/reproduce.sh`** – The glue script that ties everything together and ensures reproducibility.

## Limitations

- Only works with environments that expose a mutable `state` attribute (e.g., CartPole, MountainCar).  
- The mask training objective is a simplified surrogate of the original StateMask algorithm.  
- No external baselines (JSRL, StateMask‑R, PPO++) are included; the focus is on illustrating the RICE workflow.  
- The implementation uses a very small number of training steps to keep runtime short.

Feel free to extend this repository to add full experimental pipelines, additional environments, or baseline comparisons as needed. Happy experimenting!