# RICE – Reinforcement Learning with Explanation  
Reproduction repository for the paper *RICE: Breaking Through the Training Bottlenecks of Reinforcement Learning with Explanation*.

## Overview
This repository implements a minimal yet functional reproduction of the RICE algorithm described in the paper.  
It trains a baseline PPO agent, learns a mask network that identifies critical states, and refines the policy by  
1. resetting to a mixture of default and critical states, and  
2. encouraging exploration via Random Network Distillation (RND).

The experiments are performed on the **HalfCheetah-v3** environment (MuJoCo) but can be easily switched to other Gymnasium environments.

## Reproduction Steps
1. **Setup** – The `reproduce.sh` script installs all dependencies and runs the main script.  
2. **Baseline training** – A PPO agent is trained for 200 k timesteps.  
3. **Mask network training** – A small MLP is trained with REINFORCE to predict a binary mask that maximises the total reward plus an exploration bonus.  
4. **Refinement** – The base policy is refined for another 200 k timesteps using:
   * a mixed initial‑state distribution (probability `p` to start from a critical state),
   * an intrinsic reward from RND (`lambda`), and
   * a buffer of critical states collected during training.
5. **Evaluation** – The refined policy is evaluated over 20 episodes and the mean reward is printed.

All code is in `run.py`.  
The trained models are saved in `models/` and the final reward is logged to the console.

> **Note**  
> The implementation is intentionally lightweight to keep the repository < 1 GB and to run within the 7‑day limit.  
> For a more faithful reproduction of the paper’s results, you may increase the training timesteps, adjust hyper‑parameters, or use the other environments described in the paper.

## Usage
```bash
bash reproduce.sh
```

The script will output something like:

```
Mean reward: 2138.89 ± 3.22
```

indicating the performance of the refined policy.

## Repository Structure
```
├── reproduce.sh          # Wrapper that installs deps and runs the experiment
├── README.md             # This file
├── run.py                # Main experiment script
├── models/               # Directory where trained models are stored
└── requirements.txt      # Optional: list of pip packages (not required)
```

## License
MIT License. Feel free to adapt or extend the code for your own research.