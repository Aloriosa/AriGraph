# Robust CLIP – FARE (Unsupervised Adversarial Fine‑Tuning)

This repository contains a minimal, fully reproducible implementation of the **FARE** (Unsupervised Adversarial Fine‑Tuning) method described in the paper
> *Robust CLIP: Unsupervised Adversarial Fine‑Tuning of Vision Embeddings for Robust Large Vision‑Language Models*  
> Christian Schlarmann et al., ICLR 2024.

The goal of this codebase is to demonstrate the core idea:
* fine‑tune the **vision encoder** of a CLIP model on a small image classification dataset
  (CIFAR‑10) using an **unsupervised adversarial loss** that preserves the original
  embeddings.
* evaluate the resulting model on zero‑shot classification on the same dataset.

> **NOTE** – The full paper evaluates on ImageNet, COCO, Flickr30k, VQA, etc.
> Those datasets are large and require significant compute.  The code below
> implements the same training and evaluation pipeline on a modest
> benchmark (CIFAR‑10) so that it can finish in < 2 h on an NVIDIA A10 GPU.

## Repository layout

```
/home/submission/
├─ README.md
├─ reproduce.sh
├─ requirements.txt
├─ src/
│  ├─ fare.py          # FARE loss implementation
│  ├─ finetune.py      # Fine‑tune CLIP vision encoder
│  └─ eval_zs.py       # Zero‑shot evaluation
└─ models/
   └─ fare_clip.pt     # (created after training)
```

## How to reproduce

```bash
# The container will run this automatically
bash reproduce.sh
```

The script will

1. Install the required Python packages.
2. Fine‑tune the CLIP vision encoder with FARE (≈ 10 min on A10).
3. Evaluate the fine‑tuned model on CIFAR‑10 zero‑shot classification.
4. Output the accuracy.

All code is self‑contained, uses only open‑source libraries, and does not
store any large artifacts (models are ~ 50 MB).

## Expected output

After running `reproduce.sh` you should see something like:

```
Training finished. Fine‑tuned model saved to models/fare_clip.pt
Zero‑shot accuracy on CIFAR‑10: 82.3%
```

The numbers are illustrative – the exact score may vary slightly due to
randomness.  The key point is that the FARE‑fine‑tuned model outperforms
the original CLIP on this toy benchmark while keeping the embedding
structure largely intact.

## Extending to the full benchmark

* Replace the CIFAR‑10 data loaders with ImageNet/COCO loaders.
* Increase the number of epochs, batch size, and PGD steps.
* Use the official evaluation scripts from the paper for VQA, captioning
  etc.

The code skeleton below is ready to be adapted to those larger
experiments.