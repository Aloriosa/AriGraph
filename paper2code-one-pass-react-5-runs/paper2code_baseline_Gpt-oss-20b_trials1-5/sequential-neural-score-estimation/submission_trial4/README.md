# Sequential Neural Posterior Score Estimation (TSNPSE) Toy Implementation

This repository contains a lightweight Python implementation of the **Sequential Neural Posterior Score Estimation (TSNPSE)** algorithm described in the paper *"Sequential Neural Score Estimation: Likelihood-Free Inference with Conditional Score Based Diffusion Models"*.

> **Note**: This is a *toy* implementation designed to demonstrate the core ideas of TSNPSE on a very simple 2‑dimensional simulation model. It is **not** aimed at reproducing the full benchmark results presented in the paper. However, the code structure mirrors the main components of the original algorithm:
> 
> * forward and reverse diffusion processes (variance‑exploding SDE)
> * conditional score network (MLP)
> * sequential rounds with truncated proposals
> * training via denoising score matching
> * posterior sampling using the probability flow ODE

## Repository Layout

```
.
├── reproduce.sh                # Reproduction script
├── README.md                   # This file
├── data/
│   └── observed_data.npy       # Toy observed data (generated if missing)
├── outputs/
│   └── posterior_samples.npy  # Posterior samples produced by the script
├── src/
│   ├── __init__.py
│   ├── utils.py                # Utility functions (simulator, SDE, etc.)
│   ├── model.py                # MLP score network
│   ├── diffusion.py            # Forward / reverse diffusion helpers
│   └── train.py                # Main training & sampling loop
```

## How to Run

1. **Make the reproduction script executable**:
   ```bash
   chmod +x reproduce.sh
   ```

2. **Execute the script**:
   ```bash
   ./reproduce.sh
   ```

   The script will:
   * create a virtual environment and install dependencies
   * generate a toy observed data point if not already present
   * train a TSNPSE model for two sequential rounds
   * generate posterior samples and save them to `outputs/posterior_samples.npy`

3. **Inspect the results**:
   ```bash
   python - <<'PY'
   import numpy as np
   samples = np.load('outputs/posterior_samples.npy')
   print('Posterior samples shape:', samples.shape)
   print('Posterior mean estimate:', samples.mean(axis=0))
   PY
   ```

## What was achieved

- **Implementation**: A minimal but functional TSNPSE pipeline.
- **Training**: The model learns to approximate the posterior of a toy 2‑D Gaussian mixture model.
- **Sampling**: Posterior samples are generated via the probability flow ODE.
- **Reproducibility**: The `reproduce.sh` script can be run on any machine with Python 3.8+ and a CUDA‑capable GPU (if available). All dependencies are installed automatically.

## Limitations

- The toy model is intentionally simple; it does **not** match the complexity of the benchmarks in the paper.
- The diffusion SDE uses only a small number of discretisation steps (10) to keep runtime short.
- The sequential component uses only two rounds and a very coarse truncated proposal (simple rejection sampling).

Feel free to extend the code to larger models, more rounds, or more sophisticated SDEs (e.g., variance‑preserving).