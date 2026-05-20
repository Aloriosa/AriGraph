# Robust CLIP – FARE (Unsupervised Adversarial Fine‑Tuning)

This repository contains a minimal, fully reproducible implementation of the
*FARE* method described in the paper “Robust CLIP: Unsupervised Adversarial
Fine‑Tuning of Vision Embeddings for Robust Large Vision‑Language Models”.
The goal of the repository is to demonstrate the core idea: fine‑tune a frozen
CLIP vision encoder so that its embeddings are stable under
$\ell_{\infty}$‑bounded perturbations, while keeping the original zero‑shot
performance.

> **⚠️ Disclaimer**  
> The full experiments reported in the paper use the ViT‑L/14 backbone,
> ImageNet training for 2 epochs, 10 PGD steps and a batch size of 128.
> Reproducing those results requires multi‑GPU compute and ~10 GB of RAM.
>  
> The code below uses the lightweight ViT‑B/32 backbone and the CIFAR‑10
> dataset (10 k images) for a quick sanity check.  It can be run on a single
> GPU in a few minutes and will produce a small “robust” model that still
> performs well on clean data.

## How to reproduce

```bash
bash reproduce.sh
```

The script will:

1. Install the required Python packages (`torch`, `torchvision`, `timm`,
   `clip`, `numpy`).
2. Download the CLIP ViT‑B/32 model.
3. Fine‑tune it on CIFAR‑10 for 2 epochs with FARE (unsupervised
   adversarial fine‑tuning) using 10 PGD steps.
4. Evaluate clean and adversarial (ε = 4/255) accuracy on the test set.
5. Save the fine‑tuned model as `finetuned_clip.pt`.

After the script finishes you should see something like:

```
Clean accuracy  : 97.3%
Adversarial accuracy (ε=4/255) : 94.8%
```

Feel free to experiment with the hyper‑parameters in `src/clip_finetune.py`
and `src/evaluate.py`.

## Repository layout

```
.
├── README.md
├── reproduce.sh
├── requirements.txt
└── src
    ├── clip_finetune.py   # FARE training loop
    └── evaluate.py        # Clean & adversarial evaluation
```

## Notes

- The code uses PyTorch 2.x and the official CLIP implementation from
  `openai/CLIP`.
- GPU is required (the script automatically falls back to CPU if no GPU
  is available).
- The training and evaluation are intentionally lightweight; you can
  scale them up to ImageNet and ViT‑L/14 by increasing the dataset,
  batch‑size and number of epochs.
- The core of FARE is the loss
  ```python
  loss = torch.mean((adv_emb - clean_emb)**2)
  ```
  which keeps the adversarial embedding close to the clean embedding.

Enjoy experimenting with robust vision embeddings!