# DPMs‑ANT Reproduction

This repository contains a lightweight implementation of the **DPMs‑ANT** method described in  
*Bridging Data Gaps in Diffusion Models with Adversarial Noise‑Based Transfer Learning*.

The goal is to demonstrate the core ideas:

1. **Similarity‑guided training** – a binary classifier distinguishes source vs. target images at a noisy timestep, guiding the diffusion model toward the target domain.  
2. **Adversarial noise selection** – a small gradient‑ascent loop finds a “worst‑case” Gaussian noise for the current UNet, speeding up convergence.  
3. **Parameter‑efficient training** – only a small set of UNet weights (the classifier and a tiny LoRA‑style adapter) are updated during fine‑tuning.

## Structure

```
├── data
│   └── target.py          # download & prepare a 10‑shot “cat” set from CIFAR‑10
├── evaluate.py            # compute LPIPS & FID between generated & target images
├── train_ants.py          # training script (classifier → UNet → generation)
├── reproduce.sh           # shell script that installs deps, runs training & eval
├── requirements.txt
└── README.md
```

## Usage

```bash
bash reproduce.sh
```

`reproduce.sh` will:

1. Install the required Python packages.  
2. Download CIFAR‑10 and extract 10 cat images as the target domain.  
3. Train a binary classifier on noisy images from the source (CIFAR‑10 train set) and target domains.  
4. Fine‑tune a pre‑trained DDPM UNet on the 10‑shot target set using similarity‑guided loss & adversarial noise selection.  
5. Generate 1 000 samples with the fine‑tuned UNet.  
6. Compute LPIPS & FID against the target set and write the results to `metrics.txt`.

Only the source code is committed – all generated images and checkpoints are stored in the `generated/` and `checkpoints/` directories, respectively, and are not part of the repository.

Feel free to tweak hyper‑parameters in `train_ants.py` if you wish to experiment further.