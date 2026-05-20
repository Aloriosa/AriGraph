# SAPG Reproduction – A Toy Implementation

This repository contains a minimal, self‑contained implementation of the **Split and Aggregate Policy Gradients (SAPG)** algorithm described in the paper *"SAPG: Split and Aggregate Policy Gradients"*.

The goal of this reproduction is **not** to achieve the state‑of‑the‑art results reported in the paper (which require large‑scale GPU simulators such as IsaacGym and thousands of parallel environments).  
Instead, we provide a **fully runnable** example that demonstrates the core ideas of SAPG on a standard continuous‑control task (`Pendulum-v1`) in the OpenAI Gymnasium library.  

## What is implemented

* **Leader–Follower architecture** (one leader policy + two follower policies)
* **Shared backbone** for all policies with a small trainable latent vector (`phi`) per policy
* **On‑policy PPO loss** for each policy
* **Off‑policy importance‑weighted loss** used only for the leader
* **Simple advantage estimation** (returns – value) and a small value network
* **Entropy regularisation** (currently disabled – can be tuned via the `ENTROPY_COEF` argument)

## How to run

The repository contains a `reproduce.sh` script.  
Running it will:

1. Install the necessary Python packages (`torch`, `gymnasium`, `numpy`).
2. Train the SAPG agent for a fixed number of iterations.
3. Save a `metrics.txt` file containing the mean return of each policy after every iteration.

```bash
bash reproduce.sh
```

The script is intentionally lightweight and should finish within a few minutes on a CPU – no GPU is required.  
If you have a CUDA‑capable GPU, the code will automatically use it.

## Expected outputs

After training, you should see a file `metrics.txt` in the repository root containing lines like

```
Iteration 0: Leader= -134.52, Follower1= -139.83, Follower2= -138.91
Iteration 1: Leader= -132.88, Follower1= -137.15, Follower2= -136.72
...
Iteration 20: Leader= -122.03, Follower1= -125.47, Follower2= -124.88
```

These numbers are **mean returns** over 10 evaluation episodes per policy.  
The leader policy typically learns faster because it leverages data from the followers, demonstrating the *split‑and‑aggregate* principle.

Feel free to experiment by changing hyper‑parameters in `train_sapg.py` or adding entropy regularisation.

## Repository structure

```
.
├── reproduce.sh          # Entrypoint script
├── README.md             # This file
├── requirements.txt      # Python dependencies
└── train_sapg.py         # Core implementation
```

No heavy data or checkpoints are stored – everything is generated on the fly during training.

Happy experimenting!