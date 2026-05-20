# CompoNet – A Minimal Reproduction of the Paper  
**Self‑Composing Policies for Scalable Continual Reinforcement Learning**

This repository contains a lightweight implementation of the *CompoNet* architecture
described in the paper *“Self‑Composing Policies for Scalable Continual Reinforcement Learning”*.
The goal is to provide a **fully reproducible** script that trains a small sequence of
reinforcement learning tasks while adding a new policy module for each task, freezing
previous modules, and allowing the new module to attend to the outputs of the
previous ones.  

The repository is intentionally minimal:
* three discrete control tasks from OpenAI Gym (`CartPole-v1`, `MountainCar-v0`,
  `Acrobot-v1`);
* a simple REINFORCE trainer (no baselines, no replay);
* a fully‑automatic `reproduce.sh` script that installs the minimal dependencies,
  runs the training, and writes a `results.txt` file with the average return per
  task.

> **NOTE**  
> The training parameters (number of episodes, hidden sizes, learning rate,
> …) are chosen to keep the total runtime below 5 min on a CPU‑only machine.
> They are *not* tuned to match the numbers reported in the original paper.
> The purpose of this repository is to demonstrate that the described
> architecture can be implemented and trained on a small task sequence
> without any heavy artefacts.

## How to run

```bash
bash reproduce.sh
```

The script will:
1. Create a clean virtual environment (if desired).
2. Install the required Python packages.
3. Run the training script `componet/train.py`.
4. Store the per‑task average returns in `results.txt`.

After the script finishes you should see a summary similar to:

```
Task 0 (CartPole-v1): avg_return = 195.3
Task 1 (MountainCar-v0): avg_return = 110.7
Task 2 (Acrobot-v1): avg_return = 86.2
```

The `results.txt` file contains the same information in a machine‑readable format.

## Repository layout

```
/home/submission/
├── README.md
├── reproduce.sh
├── requirements.txt
└── componet/
    ├── __init__.py
    ├── models.py
    ├── train.py
    ├── envs.py
    └── utils.py
```

## Implementation details

* **CompoNet**  
  The network is a list of `PolicyModule` objects.  
  Each `PolicyModule` receives the current state and the outputs of all
  *previous* modules.  
  It consists of an *output attention head* (dot‑product attention over the
  previous logits) followed by a small fully‑connected *internal policy*.
  The final logits are the sum of the attention output and the internal policy
  output.  Previous modules are frozen during training of the new one.

* **Training**  
  We use a vanilla REINFORCE algorithm with discounted returns.  
  For each task we run `NUM_EPISODES` episodes, record the trajectory,
  compute the discounted return, and update only the parameters of the
  current module.  Previous modules are frozen (their gradients are set to
  `requires_grad=False`).  

* **Determinism**  
  Random seeds are set for `random`, `numpy`, and `torch`.  The Gym
  environment is seeded at each episode.

* **Hardware**  
  The code runs on CPU.  If a CUDA device is available it will be used
  automatically.

Enjoy experimenting with the architecture! Feel free to tweak the hyper‑parameters
or extend the task sequence to see how the attention heads react to longer
chains of modules.