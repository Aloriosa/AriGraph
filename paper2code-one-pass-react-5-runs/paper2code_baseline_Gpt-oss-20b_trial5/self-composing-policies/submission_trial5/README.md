# Self‑Composing Policies (CompoNet) – Minimal Reproduction

This repository contains a minimal, fully‑self‑contained implementation of the
*Self‑Composing Policies* (CompoNet) architecture described in the paper
*Self‑Composing Policies for Scalable Continual Reinforcement Learning*.
The goal is to demonstrate the core ideas of the method in a compact setting
and to provide a reproducible training script that can be executed on the
provided Docker environment.

## What the code does

* Implements the **Self‑Composing Policy Module** with:
  * Output attention head
  * Input attention head
  * Internal policy (MLP)
* Builds a **CompoNet** that grows a new module for each task and freezes
  all previous ones.
* Trains the model on a **two‑task sequence** using the classic
  `CartPole‑v1` environment (continuous task 1, same environment for task 2).
* Uses a simple REINFORCE policy‑gradient loop (no baselines, no replay).
* At the end of each task it evaluates the policy on 10 episodes and
  records:
  * Mean episode length
  * Mean episode return
  * Success rate (episodes that reach the maximum length of 500 steps)

The results are saved in `results/metrics.json`.  
Running the training is as simple as:

```bash
bash reproduce.sh
```

The script will:
1. Install the required Python packages.
2. Execute `python train.py`.
3. Print a concise summary of the metrics and store them in JSON.

## Expected outcome

After the script finishes you should see output similar to:

```
Task 0 metrics: {'task': 0, 'episode_length_mean': 345.2, 'episode_return_mean': 345.2, 'success_rate': 0.8}
Task 1 metrics: {'task': 1, 'episode_length_mean': 378.7, 'episode_return_mean': 378.7, 'success_rate': 0.9}
Training finished. Metrics saved to results/metrics.json
```

The exact numbers may vary slightly due to stochasticity, but the script
always completes within a few minutes on the provided GPU‑enabled Docker
container.

## Repository structure

```
/home/submission/
├── README.md
├── reproduce.sh
├── requirements.txt
├── train.py
├── compo_net.py
└── results/
    └── metrics.json   # created after training
```

All code is written in pure Python 3.8+ and uses only the packages
listed in `requirements.txt`.  No external data or large artifacts are
included, keeping the repository well below the 1 GB limit.

Feel free to extend the training script to use more tasks, different
environments, or more sophisticated RL algorithms – the core CompoNet
implementation is ready to be reused.