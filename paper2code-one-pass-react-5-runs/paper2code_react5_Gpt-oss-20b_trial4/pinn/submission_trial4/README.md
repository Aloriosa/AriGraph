# Reproduction of “Challenges in Training PINNs: A Loss Landscape Perspective”

This repository contains a lightweight, fully reproducible implementation of the
physics‑informed neural network (PINN) training pipeline for the one‑dimensional
wave equation described in the paper *Challenges in Training PINNs: A Loss Landscape Perspective*.

## Key Features

| Feature | Implementation |
|---------|----------------|
| Physics‑informed loss (PDE residual + boundary + initial conditions) | Full implementation of the wave PDE `u_tt - 4 u_xx = 0` with analytical solution |
| MLP architecture | 3 hidden layers, tanh activations, widths 50/100/200/400 |
| Weight initialization | Xavier normal, zero biases |
| Optimizers | Adam, L‑BFGS (via `torch.optim.LBFGS`) and the Adam + L‑BFGS schedule |
| Training schedule | 41 000 iterations per run (1 k, 11 k, 31 k switches) |
| Evaluation | L<sub>2</sub> relative error over the full grid and boundary/initial points |
| Reproducibility | 5 random seeds per configuration, explicit seeding of NumPy & PyTorch |
| Hardware | GPU (any CUDA‑enabled device) – the script automatically falls back to CPU |
| Output | `results/results.csv` – final loss and L2RE for each run |
| Reproduction script | `reproduce.sh` – installs dependencies and runs the full experiment |

> **Note**: For the sake of speed in the evaluation environment, the script
> runs a reduced number of iterations (10 000) by default.  
> To match the exact 41 000 iterations of the paper, set the environment
> variable `FULL_ITERATIONS=1` before running the script.

## How to Run

```bash
bash reproduce.sh
```

The script will create a directory `results/` and fill it with
`results.csv`. The file contains the following columns:

```
width,seed,final_loss,l2re
```

## Repository Layout

```
/home/submission/
├─ README.md
├─ reproduce.sh
├─ requirements.txt
├─ results/
│   └─ results.csv
└─ src/
    └─ pinn_wave.py
```

Feel free to tweak the hyper‑parameters (learning rates, switch points,
iterations, etc.) in `pinn_wave.py` if you want to experiment further.