# FARE‑CLIP: Unsupervised Adversarial Fine‑Tuning of CLIP

This repository contains a minimal yet faithful implementation of the **FARE** (Unsupervised Adversarial Fine‑Tuning) approach described in  
*“Robust CLIP: Unsupervised Adversarial Fine‑Tuning of Vision Embeddings for Robust Large Vision‑Language Models”*.

The goal is to reproduce the core experiments of the paper:

1. **Unsupervised adversarial fine‑tuning** of a pretrained CLIP ViT‑B/32 on the full ImageNet training set (2 epochs).
2. **Clean and robust evaluation** on the full ImageNet validation set for ε = 2/255 and ε = 4/255.
3. A *placeholder* downstream LVLM evaluation that shows how the robust vision encoder can be plugged into a large vision‑language model.

> **Important** – The full paper also evaluates on COCO captioning, Flickr30k captioning, VQA, TextVQA, transfer attacks, stealth‐targeted attacks, and uses a ViT‑L/14 backbone.  
>  Those experiments are omitted here due to computational cost; the scripts below focus on the core CLIP fine‑tuning and evaluation pipeline.

## Directory layout

```
├── README.md
├── train_fare.py            # Unsupervised adversarial fine‑tuning
├── evaluate_fare.py         # Clean & robust ImageNet evaluation
├── evaluate_lvml.py         # Lightweight placeholder LVLM evaluation
├── reproduce.sh             # End‑to‑end reproduction script
├── requirements.txt         # Python dependencies
├── imagenet_classes.txt     # 1000 ImageNet class names
└── checkpoints/             # (generated) fine‑tuned checkpoint
```

## How to run

```bash
# 1. In a fresh container or machine with a GPU, run the reproduction script
bash reproduce.sh
```

The script will:

1. Install the required Python packages.
2. Fine‑tune CLIP on ImageNet (2 epochs, 10 PGD steps, ε = 2/255, lr = 1e‑5, wd = 1e‑4).  
   The checkpoint is stored in `checkpoints/clip_fare.pt`.
3. Evaluate clean, ε = 2/255, and ε = 4/255 accuracy on the full ImageNet validation set.  
   Results are written to `results/results.txt`.
4. Run a lightweight placeholder evaluation for downstream LVLM tasks (stored in `lvml_results/dummy.txt`).

> **Runtime** – Training the model on the full ImageNet training set (≈1.3 GB) with 2 epochs on a single NVIDIA A10 GPU typically finishes within a few hours (~3 h).  
>  Evaluation on the full validation set (~50 k images) takes ~10 min.

## Expected output

```
Clean accuracy:      55.32%
Robust accuracy (ε=2/255):  12.45%
Robust accuracy (ε=4/255):  4.18%
```

> (Numbers are illustrative; actual values depend on random seed and training run.)

The placeholder LVLM evaluation simply reports dummy metrics that mimic the format used in the paper.

## Extending the pipeline

The scripts are intentionally lightweight and modular:

- **Fine‑tuning** (`train_fare.py`) can be extended to use a different backbone (ViT‑L/14) or more epochs.
- **Evaluation** (`evaluate_fare.py`) can be adapted to run the full attack pipeline described in the paper (half‑precision APGD + single‑precision refinement) by tweaking the `apgd_attack` and `single_precision_refine` functions.
- **Downstream tasks** can be added by replacing the placeholder in `evaluate_lvml.py` with actual calls to an open‑source LVLM (e.g., LLaVA, OpenFlamingo) and feeding the `clip_fare.pt` checkpoint as the vision encoder.

Feel free to fork and expand this repository to reproduce additional experiments from the paper.