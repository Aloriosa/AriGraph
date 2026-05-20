# Reproduction of *Challenges in Training PINNs: A Loss Landscape Perspective*

This repository contains a lightweight implementation of the experiments reported in  
**Pratik Rathore, Weimu Lei, Zachary Frangella, Lu Lu, Madeleine Udell**  
“Challenges in Training PINNs: A Loss Landscape Perspective” (ICML 2024).

## Overview

The code reproduces the following key findings:

| Experiment | What is reproduced |
|------------|--------------------|
| **Loss vs. L2RE** | Table 1 – Adam, L‑BFGS, Adam + L‑BFGS (1 k/11 k/31 k), and NysNewton‑CG (NNCG). |
| **Conditioning** | Spectral density of the Hessian before and after L‑BFGS. |
| **Low‑loss regime** | L2RE vs loss for the three PDEs at very small residual values. |
| **Optimizer switch** | Adam → L‑BFGS at 1 k, 11 k, or 31 k iterations. |
| **Hardware** | Single NVIDIA A10 GPU (CUDA 11.8) – code runs on any CUDA‑enabled GPU. |

Running `bash reproduce.sh` will:

1. Install the required dependencies (PyTorch 2.0.0, NumPy, tqdm, etc.).  
2. Execute the full training pipeline for all PDEs, network widths, optimizers, and random seeds.  
3. Store the loss and L2RE trajectories in `results/`.  
4. Compute and save a short diagnostic of the Hessian spectral density (top‑10 eigenvalues) for the final iterate.  
5. Print a summary table comparable to the one in the paper.

> **Important** – The results are deterministic up to the stochastic sampling of residual points.  
> For reproducibility, the random seed is fixed per experiment (5 different seeds are used as in the paper).

## Repository Layout

```
├── reproduce.sh
├── README.md
├── experiments/
│   ├── main.py          # orchestrates the experiments
│   ├── config.py        # hyper‑parameters & grid search
│   ├── models.py        # MLP definition
│   ├── pinn.py          # PDE definitions & loss
│   ├── utils.py         # metrics & helpers
│   └── hessian.py       # spectral analysis
└── results/             # automatically generated during the run
    ├── <pde>_<width>_<opt>_<seed>.json
    └── summary.txt
```

The code is self‑contained, does not ship any heavy artifacts, and runs within the 7‑day limit on a single GPU.

---

## Running the Experiments

```bash
bash reproduce.sh
```

After the script finishes, inspect the `results/` directory.  Each experiment is stored as a JSON file with the following keys:

- `loss`: list of loss values per iteration.  
- `l2re`: list of L2 relative errors per iteration.  
- `hessian_top10`: top‑10 eigenvalues of the final Hessian.  
- `hessian_condition`: ratio of the largest to smallest of the top‑10 eigenvalues.  
- `switch_point`: iteration at which Adam → L‑BFGS switch occured (if applicable).  

The `summary.txt` file contains a human‑readable table summarizing the final loss and L2RE for each configuration, mirroring the paper’s Table 1.

---

## Implementation Details

* **PDEs** – Convection, Wave, and Reaction ODE are implemented exactly as described in the paper (Appendix A).  
* **Network** – 3‑layer MLP, tanh activations, widths `{50, 100, 200, 400}`. Xavier normal initialization, zero biases.  
* **Training** – 41 000 iterations; Adam learning rates are chosen by a grid search over `{1e‑5, 1e‑4, 1e‑3, 1e‑2, 1e‑1}`; L‑BFGS uses lr = 1.0, memory = 100, and strong Wolfe line search.  
* **Adam+L‑BFGS** – Adam for the first `switch` iterations (1 k, 11 k, or 31 k) then L‑BFGS for the remainder.  
* **NNCG** – After Adam+L‑BFGS, a damped Newton step is computed via Conjugate Gradient with Hessian–vector products.  
* **Hessian analysis** – Top‑10 eigenvalues are estimated with power iteration (10 iterations each).  

Feel free to tweak hyper‑parameters in `experiments/config.py` if you wish to explore further.

---

## Acknowledgements

The code follows the experimental protocol described in the paper, and uses only publicly available libraries.  The authors of the paper are not affiliated with this repository.