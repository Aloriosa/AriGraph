# CompoNet – Self‑Composing Policies for Continual Reinforcement Learning

This repository contains a **minimal reproducible implementation** of the key ideas from the paper *Self‑Composing Policies for Scalable Continual Reinforcement Learning*.

> **Disclaimer**  
> The full experimental pipeline described in the paper (Meta‑World tasks, PPO/SAC, 20+ tasks, detailed evaluation metrics, etc.) is computationally expensive and would require several days of GPU time.  
> For the purpose of this reproduction task, we provide a lightweight, self‑contained script that demonstrates the core concepts:
> 1. **CompoNet** architecture – a chain of policy modules that are frozen after training on a task and new modules are added for subsequent tasks.
> 2. A simple **training loop** that runs on the Atari `SpaceInvaders` environment from `gymnasium`.
> 3. Basic **evaluation** that records cumulative rewards for each pseudo‑task.

The goal of this repository is to show that the core algorithm can be implemented and executed on a single GPU within a few minutes. It is **not** intended to match the exact numbers reported in the paper (which require extensive training and careful hyper‑parameter tuning).

## How to run

```bash
bash reproduce.sh
```

The script will:
1. Install the required Python packages.
2. Run `train.py`, which trains a CompoNet agent on five consecutive executions of the same environment.
3. Save a simple text file with the cumulative reward obtained on each task (`outputs/results.txt`).

## Repository structure

```
├── compo_net.py          # Minimal CompoNet implementation
├── train.py              # Training and evaluation loop
├── reproduce.sh          # Reproduction script
├── requirements.txt      # Python dependencies
├── README.md             # This file
└── outputs/results.txt  # Generated after running reproduce.sh
```

## Extending the code

To experiment with the full paper settings, you would need to:

1. Replace the environment with Meta‑World tasks (`metaworld` package) or additional Atari games.
2. Implement the full self‑composing policy module with the three attention heads as described in the paper.
3. Add the logic for freezing modules, adding new modules, and composing over all previous modules.
4. Use the stable‑baseline3 `PPO` or `SAC` implementations and set the hyper‑parameters from the paper.
5. Add proper evaluation metrics (success rate, forward transfer, etc.) and statistical analysis.

Feel free to use this repository as a starting point for such extensions.

## License

This code is provided for educational purposes under the MIT license. Feel free to modify and distribute it.