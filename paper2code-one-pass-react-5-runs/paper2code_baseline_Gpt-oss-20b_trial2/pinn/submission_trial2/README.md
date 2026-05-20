# Reproduction of “Challenges in Training PINNs: A Loss Landscape Perspective”

This repository contains a **minimal, self‑contained** implementation of the key ideas from the paper
*“Challenges in Training PINNs: A Loss Landscape Perspective”*.
The goal is to show how different optimizers (Adam, L‑BFGS, and a combined Adam+L‑BFGS) affect the
training of physics–informed neural networks (PINNs) on a small set of benchmark PDEs
(convection, reaction, and wave).

> **NOTE**  
> The implementation below is *not* a full reproduction of all plots and numerical experiments
> presented in the paper. Instead, it provides a lightweight, reproducible workflow that
> demonstrates the essential training pipeline and the relative performance of the three
> optimisation strategies.  
> Running the repository will produce a small set of CSV files containing the loss value
> and the L2 relative error (L2RE) for each experiment.

## Repository layout

```
.
├── README.md
├── requirements.txt
├── reproduce.sh          # The entry point used by the grading script
├── train_pinn.py         # Main training script
├── models.py             # Simple MLP definition
├── utils.py              # Data generation, PDE operators, loss, evaluation
└── results/              # Generated after running reproduce.sh
```

## How to run

The grading environment automatically executes `bash reproduce.sh` from the root of the
repository. The script performs the following steps:

1. Installs the required Python packages (`torch` and `numpy`).
2. Runs `train_pinn.py` which:
   - Trains a PINN on each of the three PDEs, for four network widths
     (50, 100, 200, 400) and five random seeds.
   - Uses three optimisers: Adam, L‑BFGS, Adam+L‑BFGS.
   - Stores the final training loss and the L2 relative error on a dense evaluation grid
     in CSV files under the `results/` directory.
3. Prints a short summary of the results to the console.

All files are pure source code – no heavy artefacts are committed to the repository.

## Expected artefacts

After running `reproduce.sh` you should find the following directories:

```
results/
├── convection/
│   ├── adam/
│   │   ├── width_50_seed_0.csv
│   │   ├── …
│   ├── lbfgs/
│   └── adam_lbgfs/
├── reaction/
│   ├── adam/
│   ├── lbfgs/
│   └── adam_lbgfs/
└── wave/
    ├── adam/
    ├── lbfgs/
    └── adam_lbgfs/