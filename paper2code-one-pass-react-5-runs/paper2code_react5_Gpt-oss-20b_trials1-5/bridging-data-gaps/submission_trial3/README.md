# DPMs‑ANT Toy Reproduction

This repository contains a **minimal, self‑contained** implementation of the
*“Bridging Data Gaps in Diffusion Models with Adversarial Noise‑Based Transfer Learning”* paper.
It demonstrates the core ideas of the paper in a toy setting:

* 10‑shot target dataset (10 random images from the CIFAR‑10 *cat* class).
* A pretrained DDPM (`google/ddpm-cifar10-32`) is fine‑tuned on this target set.
* A small CNN classifier is trained to distinguish noised target vs. source images.
* Fine‑tuning uses a similarity‑guided loss (Eq. (5)) and an inner adversarial
  noise maximisation loop (Eq. (6)).
* After training, 500 samples are generated and the FID against the 10‑shot
  target set is reported.

The goal is **not** to match the state‑of‑the‑art scores of the paper, but to
show a complete, reproducible training and evaluation pipeline that can run
inside the 7‑day budget on an NVIDIA A10 GPU (or on CPU).

## Quick start

```bash
# Assuming the repository is in /home/submission
bash reproduce.sh
```

The script will

1. Install dependencies,
2. Train the model (`train.py`) – about 15 min on an A10,
3. Generate samples and compute FID (`evaluate.py`),
4. Save checkpoints under `checkpoints/`
   and generated images under `samples/`.

## Repository structure

```
.
├── train.py          # Training script (full pipeline)
├── evaluate.py       # Sample generation + FID evaluation
├── reproduce.sh      # Convenience script
├── README.md
├── .gitignore
└── checkpoints/      # Saved model checkpoints (generated after training)
```

## Implementation details

### Data

* CIFAR‑10 is used as both source and target domain.
* The target class is *cat* (label 3) and consists of 10 images.
* The source dataset is the rest of the CIFAR‑10 training set
  (≈49 000 images).

### Model

* **UNet**: `google/ddpm-cifar10-32` from HuggingFace Diffusers.
* **Classifier**: lightweight convolutional network (3 conv layers,
  batch‑norm, ReLU, max‑pool + FC).

### Loss

* **Similarity‑guided loss**:  
  `L = ||ε - ε̂ - γ σ² ∇_{x_t} log p_φ(target | x_t)||²`
  where `ε̂` is the UNet prediction,
  `γ = 5.0` is the guidance weight, and `σ²` comes from the scheduler.
* **Adversarial noise**: inner loop maximizing
  `||ε - ε̂||²` for `J = 10` steps with step size `ω = 0.02`.

### Training

* 20 epochs, batch size = 32, LR = 1e‑5 for the UNet.
* The classifier is trained for 5 epochs on a mix of noised
  source and target images.

### Evaluation

* 500 samples are generated using the full DDPM sampling schedule
  (1000 timesteps).
* FID is computed against the 10‑shot target set using
  `torchmetrics.image.fid.FrechetInceptionDistance`.

## Limitations

* This is a toy reproduction; the target dataset is synthetic
  (random noise) and the evaluation is performed on a very small
  reference set, so FID values are not directly comparable to the
  paper.
* The code focuses on clarity and reproducibility rather than
  speed or memory efficiency.

## License

MIT License.  See `LICENSE` for details.