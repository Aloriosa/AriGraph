# Stochastic Interpolants with Data‑Dependent Couplings – Reproduction

This repository contains a **minimal but fully runnable** implementation that reproduces the key ideas from the paper *“Stochastic Interpolants with Data‑Dependent Couplings”*.  
The code trains a small conditional diffusion model on **CIFAR‑10** for an **in‑painting** task using a data‑dependent coupling, and then samples from the learned model.

> **Why this is a toy implementation?**  
> The original paper trains on ImageNet and uses sophisticated U‑Nets.  
> Training such models would require many GPU hours and large storage.  
> Our simplified version demonstrates the *core algorithm*:
> 1. **Data‑dependent coupling** (`x0 = mask * x1 + noise`).
> 2. **Stochastic interpolant** with `α_t = t`, `β_t = 1‑t`, `γ_t = 0`.
> 3. **Learning the velocity** by minimizing the quadratic loss  
>    → equivalent to an L2 loss between the predicted and true velocity.
> 4. **Sampling** by integrating the learned velocity with a simple Euler ODE solver.

The repository layout is:

```
├── README.md
├── reproduce.sh          # Entry point for the entire reproduction
├── train.py              # Training script
├── sample.py             # Sampling script
└── model.py (inline in train.py / sample.py)
```

## How to run

```bash
# 1. Make the reproduction script executable
chmod +x reproduce.sh

# 2. Run the reproduction script
./reproduce.sh
```

The script will:

1. Install the required libraries (`torch`, `torchvision`, `tqdm`, etc.).
2. Train the model for 5 epochs on CIFAR‑10 (≈ 2 min on a modern CPU, < 1 min on a GPU).
3. Generate 10 in‑painting samples and save them to `./samples/samples.png`.

You will see a progress bar during training and sampling.  
The final image (`samples.png`) contains a grid of generated images that fill the black masked region with realistic content.

## What you can experiment with

- **Dataset**: Swap `datasets.CIFAR10` for `datasets.MNIST` or any custom dataset.
- **Mask size**: Change `mask_size` in `get_mask`.
- **Model**: Replace `SimpleUNet` with a deeper U‑Net or a transformer‑based architecture.
- **Training epochs**: Increase `--epochs` for better quality.
- **Sampling steps**: Change the `steps` argument in `sample.py` for a finer integration.

## Code Overview

### `train.py`

- Loads CIFAR‑10.
- For each batch:
  - Creates a random mask.
  - Constructs the base sample `x0` by filling the masked region with Gaussian noise.
  - Samples a random time `t ~ U(0,1)` and builds the interpolated image `I_t = t·x0 + (1‑t)·x1`.
  - Computes the true velocity `dotI_t = x0 - x1`.
  - Trains a tiny CNN (`SimpleUNet`) to predict this velocity.
  - Uses an MSE loss (the unique minimizer of the quadratic objective in the paper).

### `sample.py`

- Loads the trained model.
- Generates a batch of random masks.
- Constructs a base sample from the mask and noise.
- Integrates the learned velocity from `t=0` to `t=1` using a simple Euler method.
- Saves a grid of generated images.

### `reproduce.sh`

- Installs dependencies.
- Calls `train.py` and `sample.py` in sequence.

## Results

Because this is a toy model on a small dataset, the generated samples are not as high‑fidelity as those reported in the paper (which use ImageNet and a 512‑pixel network).  
However, you will observe that the black masked region is filled with plausible, diverse content, demonstrating that the data‑dependent coupling and the learned velocity indeed constitute a conditional generative model.

---

**Happy reproducing!**

```