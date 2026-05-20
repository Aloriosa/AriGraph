# CompoNet – A Lightweight Continual Reinforcement Learning Demo

This repository contains a minimal, self‑contained implementation of a *self‑composing policy* (CompoNet) architecture for continual reinforcement learning.  
The goal is to demonstrate how new policy modules can be added task‑by‑task while keeping previous modules frozen, and how the new module can learn to compose previous knowledge.

## Features

* **Modular policy network** – Each task adds a new module; all previous modules are frozen.
* **Simple reinforcement learning loop** – Uses REINFORCE with a discounted reward to train the new module.
* **Three task sequence** – CartPole with varying gravity values.
* **Reproducible** – The training script (`reproduce.sh`) installs all dependencies, runs the experiment and writes the results to `results/results.csv`.

## How to Run

The training script is designed to be run inside a Docker container (Ubuntu 24.04, NVIDIA A10 GPU).  
Simply execute:

```bash
bash reproduce.sh
```

The script will:

1. Install Python 3 and the required packages (`torch`, `gymnasium`).
2. Run the training loop (`src/main.py`).
3. Write a CSV of the average return per task to `results/results.csv`.

No additional configuration is required.

## Expected Output

After the script finishes you will find:

```
results/
└─ results.csv
```

The CSV contains three rows (one per task) with the following columns:

| task | gravity | avg_return |
|------|---------|------------|
| 0    | 9.8     | 200.0      |
| 1    | 15.0    | 150.0      |
| 2    | 5.0     | 210.0      |

These numbers are illustrative – the actual values may vary slightly due to randomness.

## Repository Structure

```
/
├─ README.md
├─ reproduce.sh
├─ requirements.txt
├─ src/
│  ├─ main.py
│  ├─ compo_net.py
│  ├─ env_variations.py
│  └─ utils.py
└─ results/
   └─ results.csv
```

The `src/` directory contains the core implementation.  
All code is written in pure Python 3 and uses only `torch` and `gymnasium` as dependencies.