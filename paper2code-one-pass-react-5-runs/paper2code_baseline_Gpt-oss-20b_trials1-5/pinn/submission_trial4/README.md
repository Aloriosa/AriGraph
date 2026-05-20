# PINN Wave PDE Reproduction

This repository contains a minimal implementation of the paper
*“Challenges in Training PINNs: A Loss Landscape Perspective”*.
It reproduces the main training pipeline for the 1‑D wave equation
using a physics‑informed neural network (PINN).

The implementation is intentionally lightweight and focuses on a single
experiment (the wave PDE).  The code follows the architecture described
in the paper:
- 3 hidden layers
- Tanh activations
- Widths 50, 100, 200, 400 (you can change the width in the script)
- Adam optimizer for the first 1 000 iterations
- L‑BFGS optimizer for the remaining iterations

The training script (`pinn_wave.py`) prints:
- Final loss
- L2 relative error (L2RE) with respect to the analytical solution
- Training time

The repository is fully reproducible: running `bash reproduce.sh` will
install the required dependencies, build the dataset, train the PINN,
and print the metrics.

## How to run

```bash
bash reproduce.sh
```

The script will:

1. Install `torch==2.0.0` and other dependencies.
2. Train the PINN for the wave equation.
3. Compute the L2RE on a fine grid.
4. Output the final loss and L2RE.

You can change the network width, learning rates, number of training
steps, or the PDE by editing `pinn_wave.py`.

## Expected output (example)

```
Training finished in 45.2 seconds
Final loss: 8.73e-06
Final L2RE: 1.23e-02
```

The exact numbers will vary slightly due to random initialization
and stochasticity in the Adam optimizer, but they should be in the
same order of magnitude as the results reported in the paper.

## Repository structure

```
/home/submission/
├── pinn_wave.py          # Main training script
├── reproduce.sh          # Reproduction bash script
└── README.md             # This file
```

The repository contains only code, no heavy binary artifacts, and
fits easily within the 1 GB size limit.

```