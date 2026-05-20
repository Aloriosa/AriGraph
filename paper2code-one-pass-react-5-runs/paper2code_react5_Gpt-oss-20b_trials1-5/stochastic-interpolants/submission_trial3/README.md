# Stochastic Interpolants with Data‑Dependent Couplings – Minimal Reproduction

This repository contains a lightweight implementation of the core ideas from  
*Stochastic Interpolants with Data‑Dependent Couplings* (Albergo et al., 2024).  
The code reproduces an **image in‑painting** experiment on CIFAR‑10, a toy
proxy for the ImageNet experiments described in the paper.  The main
components are:

1. **Data‑dependent coupling** – the base density is conditioned on the
   target image via a mask.
2. **Stochastic interpolant** – the process  
   `I_t = (1‑t)·x₀ + t·x₁` (γₜ = 0).
3. **Flow‑matching objective** – a simple quadratic loss that learns the
   velocity field `b_t(x, ξ)` (here `ξ` is the mask).
4. **Sampling** – a forward Euler integration of the probability‑flow ODE.

The training and sampling scripts are fully self‑contained and can be run
on any machine with PyTorch (CUDA is optional).  No external datasets
larger than CIFAR‑10 are required.

## How to run

```bash
# The reproduce script sets up a virtual environment, installs
# dependencies, trains the model and generates samples.
bash reproduce.sh
```

After the script finishes you will find:
* `samples/` – a handful of generated in‑painted images.
* `model.pt` – the trained velocity model.

> **Note**  
> The experiments are intentionally lightweight.  For a faithful
> replication of the paper’s ImageNet results you would need a large
> GPU, a full ImageNet training loop, and the exact hyper‑parameters
> reported in the paper.  This repository demonstrates the core
> algorithmic ideas and can be extended to larger datasets.

## File structure

```
├── README.md
├── requirements.txt
├── reproduce.sh
├── train.py
├── sample.py
├── utils.py
└── models.py
```

---

### Implementation details

* **Mask** – a binary mask of shape `(B, 1, 32, 32)` is sampled per
  image with probability `p=0.3` per pixel.
* **Base density** – `x₀ = mask ∘ x₁ + (1‑mask) ∘ 𝒩(0, I_d)`.
* **Interpolator** – `α_t = 1‑t`, `β_t = t`, `γ_t = 0`.  
  Consequently `Ẋ_t = -x₀ + x₁`.
* **Velocity model** – a very small U‑Net that takes as input
  `(x_t, mask, t)` and outputs a 3‑channel velocity field.
* **Training loss** –  
  `L = E[‖b̂_t(I_t) - Ẋ_t‖²]` (the cross‑term in the paper’s
  quadratic loss vanishes with `γ_t = 0`).
* **Sampling** – forward Euler with `N=1000` steps, starting from
  `x₀` and integrating the learned velocity field.

Feel free to tweak hyper‑parameters, network architecture, or dataset
to explore the method further.