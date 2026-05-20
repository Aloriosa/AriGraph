# Reproduction of “Stochastic Interpolants with Data‑Dependent Couplings”

This repository contains a **minimal, self‑contained implementation** of the core ideas from the paper *Stochastic Interpolants with Data‑Dependent Couplings*.  
The goal is to demonstrate the training of a velocity field via the simple quadratic loss
proposed in the paper and to generate samples by integrating the resulting probability
flow ODE.

## Repository Structure

```
.
├── assets/                # (optional) placeholder for images
├── reproduce.sh           # Main reproducibility script
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── train.py               # Training routine
├── sample.py              # Sampling routine
├── model.py               # Velocity network & time embedding
└── samples/               # Generated samples (created after running reproduce.sh)
```

## How to Run

```bash
bash reproduce.sh
```

The script will:

1. Install Python and the required packages (`torch`, `torchvision`, `numpy`, `pillow`).
2. Train a velocity network on the MNIST dataset for **5 epochs** (change `--epochs`
   in `train.py` if desired).
3. Generate **10 samples** by integrating the learned velocity field with a simple
   Euler solver and save them in the `samples/` directory.

## What to Expect

- **Training log**: The script prints the average training loss per epoch.
- **Generated images**: The `samples/` folder will contain PNG images that look
  like MNIST digits.  They are produced by starting from a random Gaussian vector
  and evolving it through the learned velocity field.

The implementation follows the framework described in the paper:

- **Stochastic interpolant**: \( I_t = (1-t) \, x_0 + t \, x_1 \) with  
  \( \dot I_t = -x_0 + x_1 \).
- **Velocity learning**: Minimize  
  \(\displaystyle \mathbb{E}\big[\,\|\,\hat b_t(I_t)-\dot I_t \,\|^2 \big]\),  
  which is equivalent to the square‑loss objective in the paper.
- **Sampling**: Integrate the probability flow ODE  
  \( \dot X_t = \hat b_t(X_t) \) with a fixed step size.

Feel free to modify hyper‑parameters (e.g., number of epochs, batch size, learning
rate, network size) to explore the effect on training dynamics and sample quality.

## Credits

The code is inspired by the concepts in the paper and follows the same notation as
the original authors.  It is written from scratch for clarity and reproducibility.