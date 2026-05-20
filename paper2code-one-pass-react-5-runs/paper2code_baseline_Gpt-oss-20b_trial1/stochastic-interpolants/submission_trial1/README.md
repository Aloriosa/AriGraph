# Stochastic Interpolants with Data‑Dependent Couplings  
Reimplementation of the paper *“Stochastic Interpolants with Data‑Dependent Couplings”*  
by Michael S. Albergo, Mark Goldstein, Nicholas M. Boffi, Rajesh Ranganath, and Eric Vanden‑Eijnden.

This repository contains a minimal, self‑contained implementation that demonstrates the core ideas of the paper:
* a data‑dependent coupling between a base density and the target density,
* a simple stochastic interpolant,
* a square‑loss training objective for the velocity field,
* deterministic probability‑flow sampling via an ODE solver.

The code is written in PyTorch and runs on an NVIDIA GPU (or CPU if no GPU is available).  
It trains on the MNIST dataset for a single epoch and then samples a few images using the learned model.

> **Note**  
> The full experiments from the paper (ImageNet super‑resolution, in‑painting, etc.) require large datasets and training time that are not feasible in the evaluation environment.  
> This implementation focuses on the *algorithmic skeleton* and is fully reproducible within a few minutes.

## How to run

```bash
bash reproduce.sh
```

The script will

1. Install the required Python packages.  
2. Download the MNIST dataset.  
3. Train a small velocity network for one epoch.  
4. Generate 5 samples and save them to `outputs/`.

You can inspect the generated images in `outputs/`.  
The training loss is logged to the console.

## Repository structure

```
├── reproduce.sh          # Bash script that installs dependencies and runs the training
├── requirements.txt      # Python package requirements
├── main.py               # Training and sampling script
├── models.py             # Velocity network definition
├── utils.py              # Helper functions (time embedding, interpolant)
├── outputs/              # Generated samples (created by the script)
└── README.md
```

The code follows the notation used in the paper:

* \(x_1\) – target image (MNIST digit).  
* \(x_0 = \xi \circ x_1 + (1-\xi)\circ \zeta\) – base image, where \(\xi\) is a random binary mask and \(\zeta\) is Gaussian noise.  
* \(\alpha_t = 1-t,\;\beta_t = t\) – linear interpolation.  
* \(I_t = \alpha_t x_0 + \beta_t x_1\).  
* \(\dot{I}_t = \dot{\alpha}_t x_0 + \dot{\beta}_t x_1\) with \(\dot{\alpha}_t = -1,\;\dot{\beta}_t = 1\).  
* The velocity network \( \hat{b}_t(I_t) \) is trained to minimize  
  \(\mathbb{E}\left[ \bigl\|\hat{b}_t(I_t) - \dot{I}_t\bigr\|^2 \right]\),  
  which is equivalent to the quadratic objective in the paper.

The sampling procedure integrates the ODE  
\(\dot{X}_t = \hat{b}_t(X_t)\) from \(t=0\) to \(t=1\) using the `torchdiffeq` DOPRI solver.

## Acknowledgements

The implementation is inspired by the official code of the paper
(https://github.com/interpolants/couplings) and by the diffusion‑model
implementations in `torchdiffeq` and `diffusers`.