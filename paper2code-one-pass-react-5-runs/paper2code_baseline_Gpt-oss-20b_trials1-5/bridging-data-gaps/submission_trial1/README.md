# DPMs-ANT Reproduction (Simplified)

This repository contains a lightweight, self‑contained reproduction of a diffusion model
fine‑tuning pipeline inspired by the *“Bridging Data Gaps in Diffusion Models with
Adversarial Noise‑Based Transfer Learning”* paper.  
The goal is to demonstrate that a diffusion model can be adapted to a new
domain with only a handful of images while keeping the code fully
reproducible and easily executable inside a fresh Ubuntu 24.04 LTS Docker
container with an NVIDIA A10 GPU.

> **Important**  
> 1. The implementation is *not* a faithful, full‑scale re‑implementation of
>    DPMs‑ANT.  Instead, it focuses on the core idea of *few‑shot diffusion
>    fine‑tuning* with a small adapter and a simple similarity‑guided loss.
> 2. The training is performed on the 32×32 CIFAR‑10 dataset using the
>    pre‑trained `google/ddpm-cifar10-32` model from HuggingFace’s
>    `diffusers` library.  Only 10 images are used for fine‑tuning.
> 3. The reproduction script (`reproduce.sh`) installs all dependencies,
>    fine‑tunes the model, and generates 5 samples from the adapted model.
> 4. The output artifacts are small (few kilobytes) and will be committed
>    to the repository.  No large datasets are stored.

## Repository layout

```
/home/submission/
├── README.md
├── reproduce.sh
├── requirements.txt
├── src/
│   ├── train.py          # Fine‑tuning script
│   ├── generate.py       # Sampling from the fine‑tuned model
│   └── adapter.py        # Small adapter for the UNet
├── assets/               # (Optional) small image assets
└── .gitignore
```

## How to run

```bash
# From the submission directory
bash reproduce.sh
```

The script will:

1. Install the required Python packages (PyTorch, diffusers, etc.).
2. Download the CIFAR‑10 dataset (only the first 10 training images are used).
3. Fine‑tune the pre‑trained DDPM (`google/ddpm-cifar10-32`) for 3 epochs
   with a small adapter and a similarity‑guided MSE loss.
4. Save the adapted model to `output/model.pt`.
5. Generate 5 images from the adapted model and save them to
   `output/generated/`.

After the script finishes, you should see:

```
Reproduction completed. Generated images in output/generated/
```

All outputs are deterministic because seeds are fixed (seed 42).

## Expected artifacts

- `output/model.pt` – the fine‑tuned UNet weights (≈ 8 MB).
- `output/generated/` – 5 PNG images generated from the adapted model.
- `output/metrics.txt` – simple log of training loss per epoch.

The scripts are intentionally lightweight so that the entire repository stays
well below the 1 GB limit.  The focus of this reproduction is to prove that
the overall pipeline (download → fine‑tune → generate) works correctly in a
fresh container.

---

Happy reproducing!