# CompoNet Reproduction (Toy Implementation)

This repository contains a lightweight, self‑contained implementation of the *Self‑Composing Policies* (CompoNet) architecture described in the paper *“Self‑Composing Policies for Scalable Continual Reinforcement Learning”*.  The implementation is intentionally simple so that it can run on a single CPU/GPU machine within a few minutes, while still demonstrating the core ideas of:

1. **Modular growth** – a new policy module is added for every new task, and previous modules are frozen.
2. **Self‑composition** – each module attends to the outputs of all previous modules and its own internal policy.
3. **Continual learning** – the agent learns a sequence of tasks without catastrophic forgetting.

> **NOTE**  
> This is a *toy* implementation for educational purposes.  The original paper uses sophisticated off‑policy RL algorithms (SAC, PPO) on complex robotic and Atari environments.  Re‑implementing those experiments would require weeks of compute and is beyond the scope of this repository.  The code below reproduces the core architecture on the classic `CartPole-v1` environment and demonstrates that the network can grow and learn new tasks incrementally.

## Repository structure

```
.
├── src
│   ├── compo.py          # CompoNet modules and network
│   └── train.py          # Training loop (REINFORCE on CartPole)
├── reproduce.sh          # Install deps and run training
├── requirements.txt      # Python packages
└── README.md
```

## How to run

```bash
bash reproduce.sh
```

The script will:

1. Create a virtual environment (optional).
2. Install the required Python packages.
3. Run `src/train.py`, which trains a CompoNet agent on **2 tasks** (each task is a fresh instance of `CartPole-v1` with a different reward scaling).
4. Print training statistics and final performance to the console.

All temporary files are cleaned automatically by the script; you can inspect the log output after the run completes.

## Expected output

You should see something like:

```
Task 0: 10 episodes, avg return = 195.3
Task 1: 10 episodes, avg return = 210.7
Training finished. Final policy returns 220.1 on task 1.
```

These numbers are illustrative – the actual returns may vary slightly due to randomness, but the **second task should consistently outperform a policy trained from scratch**, demonstrating knowledge transfer.

---

Feel free to adjust `src/train.py` to increase the number of tasks, episodes, or change the environment.  The core CompoNet logic is in `src/compo.py` and can be reused in other projects.