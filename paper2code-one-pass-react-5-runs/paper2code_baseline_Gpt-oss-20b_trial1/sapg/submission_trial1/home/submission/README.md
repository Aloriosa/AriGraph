# SAPG (Split & Aggregate Policy Gradients) – Toy Reproduction

This repository contains a **minimal, fully‑reproducible implementation** of the SAPG algorithm described in *“SAPG: Split and Aggregate Policy Gradients”*.  
Because the original work was evaluated on highly complex, GPU‑based simulation environments (IsaacGym) with thousands of parallel workers, we provide a **conceptual toy version** that runs on CPU and uses the classic `gymnasium` CartPole environment.  

## Repository layout

- `requirements.txt` – Python dependencies.  
- `src/` – Source code:
  - `sapg.py` – Implementation of the simplified SAPG algorithm.  
  - `env_factory.py` – Helper to create vectorized Gym environments.  
  - `train.py` – Self‑contained training script that logs progress and saves the final policy.  
- `reproduce.sh` – Bash script that installs dependencies and runs training.  

## How to run

```bash
# From the repository root
bash reproduce.sh
```

The script will:
1. Install the required Python packages.  
2. Train the toy SAPG agent for 20 k environment steps.  
3. Save the final policy and training logs in `./results/`.

After training, you can load the policy with PyTorch and evaluate it:

```python
import torch
from gymnasium import make
from src.sapg import SAPGPolicy

policy = SAPGPolicy.load("results/sapg_policy.pt")
env = make("CartPole-v1")
obs = env.reset()[0]
done = False
while not done:
    action = policy.act(obs, deterministic=True)
    obs, reward, done, _, _ = env.step(action)
```

## What this toy implementation demonstrates

- **Splitting environments** into two blocks and maintaining a *leader* and *follower* policy.  
- **Off‑policy updates** for the leader using importance‑sampling from the follower’s trajectories.  
- **Shared backbone** with per‑policy latent vectors (`phi`).  
- **Entropy regularisation** for the follower to encourage exploration.  

Although the performance on CartPole is trivial, the code structure mirrors the paper’s algorithmic flow and can be extended to more complex environments (e.g. Mujoco, IsaacGym) with minimal changes.

## Limitations

- Only **CPU** execution.  
- Very small batch size (8 parallel envs).  
- No GPU‑accelerated physics or high‑dimensional tasks.  
- The algorithmic details (e.g. multi‑step returns, clipping hyper‑parameters) are simplified for clarity.

Feel free to use this as a starting point for a full‑scale implementation.