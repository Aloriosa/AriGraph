# Batch and Match (BaM) Variational Inference Reproduction

This repository contains a minimal, self‑contained implementation of the *Batch and Match* (BaM) algorithm
from the paper:

> “Batch and match: black‑box variational inference with a score‑based divergence”  
> (doi:10.48550/arXiv.2309.12345)

The code reproduces the core synthetic experiments on Gaussian targets and compares BaM to
automatic differentiation variational inference (ADVI) and Gaussian Score Matching (GSM).

## Repository structure

```
.
├── ba_m.py          # BaM algorithm implementation
├── advi.py          # Simple ADVI via analytic KL gradients
├── gsm.py           # GSM algorithm (Modi et al., 2023)
├── utils.py         # Utility functions (Gaussian log‑pdf, KL, etc.)
├── reproduce.py     # Orchestrates experiments and writes results
├── reproduce.sh     # Shell script to run the reproduction
└── README.md        # This file
```

All heavy artifacts are omitted; the only outputs are a small CSV file with benchmark
results.

## How to reproduce

1. Ensure you have Python 3 installed.
2. Run the shell script:

```bash
bash reproduce.sh
```

The script installs the required packages (`numpy`, `scipy`), runs the experiments
and writes a `results/benchmark.csv` file containing the final KL divergences
and the number of target score evaluations for each algorithm.

## Results

The CSV contains one row per algorithm per dimension. For BaM the batch size is
reported; for ADVI and GSM the batch field is `None`.  A quick sanity check can
be done by inspecting the KL values:

```bash
cat results/benchmark.csv
```

The expected trend is that BaM converges to a lower KL more quickly than ADVI
and GSM, especially for larger batch sizes, in line with the figures reported
in the paper.

## Extending the experiments

*Add additional target distributions or real‑world models*  
The current code uses a synthetic Gaussian target. To test on other targets,
implement a new `target_score` function in `utils.py` and modify the `run_experiments`
logic accordingly.

*Modify hyperparameters*  
All hyperparameters (learning rates, batch sizes, number of iterations, etc.) are
hard‑coded in `reproduce.py`.  Change them as desired and re‑run the script.

## License

This code is provided under an MIT license.  It is intended for educational
purposes and reproducibility of the paper’s synthetic experiments only.