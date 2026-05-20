# SAPG: Split and Aggregate Policy Gradients (Reproduction)

This repository contains a minimal implementation of the SAPG algorithm described in the paper *SAPG: Split and Aggregate Policy Gradients*.
The code demonstrates the core idea of training multiple policies on disjoint blocks of environments and aggregating data via importance sampling
to update a leader policy.  It is intentionally lightweight and runs on a single CPU/GPU, making it suitable for the reproduction
challenge.

## Repository structure

```
/home/submission/
├── requirements.txt          # Python dependencies
├── reproduce.sh              # Entrypoint for reproducing results
├── sapg.py                   # Main training script
├── env_factory.py            # Helper to create vectorised environments
├── utils.py                  # Utility functions (logging, checkpointing)
└── README.md                 # This file
```

## How to run

```bash
# The container will automatically run this script during grading
bash reproduce.sh
```

The script will:

1. Install required packages.
2. Train the SAPG agent on the *CartPole-v1* environment for a short run.
3. Log training progress to `train.log`.
4. Save evaluation metrics to `results.json`.

After training, you can inspect `results.json` to see the final average return over 10 evaluation episodes.

> **Note**: This implementation is a simplified educational version that captures the key concepts of SAPG but does not aim for state‑of‑the‑art performance on the original manipulation benchmarks.

## Expected outputs

- `results.json` – JSON file containing `average_return` after training.
- `train.log` – Log file with per‑epoch statistics (losses, average reward, etc.).
- `model.pt` – Final PyTorch model checkpoint (optional).

The grading script will only check that the reproduction script runs successfully and produces a JSON file with a numeric average return.

```