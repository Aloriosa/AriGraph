# DPMs‑ANT: Diffusion Model Transfer with Adversarial Noise  

This repository contains a minimal, self‑contained implementation of the **DPMs‑ANT** algorithm described in  
*Bridging Data Gaps in Diffusion Models with Adversarial Noise‑Based Transfer Learning*.  
It demonstrates how to fine‑tune a pre‑trained DDPM (on CIFAR‑10) to a few‑shot target domain (10 images) using  
1. **Similarity‑guided training** via a binary classifier that distinguishes source vs. target noised images.  
2. **Adversarial noise selection** – a small inner gradient‑ascent loop that finds the “worst‑case” Gaussian noise per sample.  
3. A lightweight **adaptor** layer added to the UNet that learns the domain shift while keeping most of the pre‑trained weights frozen.  

## Reproduction

The `reproduce.sh` script installs dependencies, downloads the required datasets, runs the training, generates 1000 images, and prints the LPIPS and FID scores.  

```bash
bash reproduce.sh
```

All output (generated images, checkpoints, evaluation logs) will be written to `outputs/`.  
The script runs on any Ubuntu 24.04 container with an NVIDIA GPU (CUDA 12.1).  

## Expected Results

Running the script will produce:

- 1000 generated images in `outputs/generated_images/`
- `outputs/evaluation/metrics.json` containing LPIPS and FID numbers.
- Checkpoints for the fine‑tuned model and classifier.

The numbers will not match the paper exactly because we use a toy CIFAR‑10 setting, but the training pipeline and algorithmic components are faithful to the paper.  

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── reproduce.sh
├── src
│   ├── __init__.py
│   ├── train.py          # Main training script
│   ├── utils.py          # Data loaders, checkpointing, evaluation helpers
│   ├── model.py          # Diffusion model + adaptor
│   ├── classifier.py     # Binary classifier for similarity guidance
│   └── loss.py           # Loss functions (MSE + similarity term)
└── outputs
    ├── checkpoints      # Model checkpoints
    ├── generated_images
    └── evaluation
```

Feel free to change hyper‑parameters in `src/train.py` or pass them via command line; the defaults are tuned for a quick demo (~2 h on an A10 GPU).  

Enjoy experimenting with DPMs‑ANT!