# Reproduction of “Stochastic Interpolants with Data‑Dependent Couplings”

This repository implements the core ideas from the paper *Stochastic Interpolants with Data‑Dependent Couplings* and demonstrates them on two conditional image generation tasks:

1. **Image in‑painting** – random 4×4 blocks on CIFAR‑10.  
2. **Super‑resolution** – doubling the resolution of CIFAR‑10 images (32×32 → 64×64).

The implementation follows the stochastic‑interpolant framework:
* A data‑dependent coupling of the base and target densities  
  \[
  \rho(x_0,x_1)=\rho_1(x_1)\rho_0(x_0|x_1)
  \]
* A continuous‑time velocity field estimated by the square‑loss regression  
  \[
  L_b(\hat b)=\int_0^1 \mathbb{E}\!\bigl[\|\hat b(I_t)\|^2-2\dot I_t\!\cdot\!\hat b(I_t)\bigr]\,dt .
  \]
* Sampling via the probability‑flow ODE solved with a high‑order ODE solver (`torchdiffeq`).

The training and evaluation pipeline is orchestrated by `reproduce.sh`.  
After running the script, the folder `assets/` will contain the best model checkpoints, generated samples, and a text file with PSNR results.

> **Note**: The original paper uses ImageNet; here we use CIFAR‑10 for practicality and to keep the repository lightweight. The same framework applies to larger datasets.

## Installation

```bash
bash reproduce.sh
```

The script will:
1. Install dependencies (`torch`, `torchvision`, `torchdiffeq`, `tqdm`, `numpy`, `Pillow`).
2. Download CIFAR‑10.
3. Train the models for both tasks.
4. Generate samples and compute PSNR.
5. Store results in `assets/results.txt`.

Feel free to adjust hyper‑parameters in `train.py` or experiment with different datasets.