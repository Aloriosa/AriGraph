# SEMA – Self‑Expansion of Pre‑trained Models with Mixture of Adapters

This repository contains a lightweight, reproducible implementation of the **SEMA** continual‑learning framework described in  
*Self‑Expansion of Pre‑trained Models with Mixture of Adapters for Continual Learning* (Wang et al., 2023).

The implementation focuses on the core ideas:

* **Frozen Vision‑Transformer** backbone (ViT‑B/16).
* **Modular adapters** (down–up linear block) attached to every transformer block.
* **Representation descriptors** (auto‑encoders) that monitor the distribution of intermediate features.
* An **expandable weighting router** that learns a soft mixture over the adapters.
* A **self‑expansion** strategy that adds a new adapter/descriptor/router only when the
  reconstruction error of all existing descriptors at a layer exceeds a z‑score threshold.
* A simple **class‑incremental** training loop on CIFAR‑100 (10 tasks × 10 classes).

Running `bash reproduce.sh` will:

1. Install the required packages (`torch`, `torchvision`, `timm`, `tqdm`, `numpy`).  
2. Download CIFAR‑100.  
3. Train the SEMA model in a class‑incremental setting.  
4. Evaluate after every task and write the per‑task accuracy to `results.txt`.

> **NOTE** – The numbers reported by this toy implementation are *not* intended to match the paper’s benchmarks.  
> They are deliberately small so that the code can run on a single NVIDIA GPU and finish well within the 7‑day limit.

Feel free to extend the code to other datasets (ImageNet‑R, ImageNet‑A, VTAB) or to use a VAE instead of the simple AE used here.

---

## Repository layout

```
.
├── reproduce.sh          # entry point for the reproduction
├── main.py               # orchestrates training and evaluation
├── src/
│   ├── __init__.py
│   ├── models.py         # ViT backbone, adapters, descriptors, router, SEMA
│   ├── dataset.py        # CIFAR‑100 task splits
│   ├── trainer.py        # training loop
│   └── utils.py          # helpers (seeds, metrics)
├── results.txt           # per‑task accuracies
└── README.md
```

## Reproduction

```bash
bash reproduce.sh
```

The script will produce `results.txt` containing:

```
Task 1 Accuracy: ...
Task 2 Accuracy: ...
...
Average Incremental Accuracy: ...
Final Accuracy: ...
```

All code is deterministic (fixed seeds) and designed to run on any Ubuntu‑based Docker container with an NVIDIA GPU.

---