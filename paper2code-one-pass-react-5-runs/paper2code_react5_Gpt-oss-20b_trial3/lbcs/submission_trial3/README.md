# Refined Coreset Selection (RCS) – Minimal Reproduction

This repository contains a lightweight implementation that demonstrates key ideas from the paper  
**“Refined Coreset Selection: Towards Minimal Coreset Size under Model Performance Constraints”**.

> **Goal** –  
> The implementation focuses on the MNIST dataset.  
> 1. Train a small CNN on the *full* training set.  
> 2. Train the same network on a *randomly selected* coreset (size `k`).  
> 3. (Optional) Run a very small greedy refinement that tries to improve the coreset
>    by swapping one training example at a time.  
> The script prints the test accuracy of each run and the coreset size used.

All heavy data downloads and large model training are omitted to keep the repository
well below the 1 GB limit.  The code is fully reproducible and can be executed on any
Ubuntu 24.04 Docker container with an NVIDIA GPU.

## How to run

```bash
bash reproduce.sh
```

The script will:

1. Install `torch` and `torchvision`.
2. Download MNIST.
3. Train the model for the full data and the random coreset.
4. Print the results.

All results are printed to standard output.

## Repository Structure

```
├── README.md
├── reproduce.sh
└── src
    ├── __init__.py
    ├── reproduce.py   # main script
    ├── models.py      # CNN definition
    ├── dataset.py     # dataset loading utilities
    ├── trainer.py     # training & evaluation helpers
    └── utils.py       # helper functions
```

Feel free to extend the `src` modules to add the full bilevel optimisation
described in the paper.  The current implementation is a minimal, fully
reproducible prototype.