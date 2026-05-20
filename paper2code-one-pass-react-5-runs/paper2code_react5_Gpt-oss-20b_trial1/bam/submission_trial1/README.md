# BaM – Batch and Match Variational Inference

This repository contains a minimal, self‑contained implementation of the **BaM** algorithm (Batch and Match) from the paper
> *Batch and match: black‑box variational inference with a score‑based divergence*  
> Diana Cai, Chirag Modi, et al., 2024.

The goal is to reproduce the core method and demonstrate its convergence on a synthetic Gaussian target.  
The implementation follows the pseudocode in the paper and uses only NumPy, SciPy, and Matplotlib, so it runs on CPU or GPU (NumPy automatically uses available hardware).

## Repository structure

```
├── requirements.txt          # Python dependencies
├── reproduce.sh              # Full reproducibility script
├── README.md
├── src/
│   ├── baum.py               # BaM algorithm implementation
│   └── utils.py              # Helper functions (KL, matrix sqrt)
├── examples/
│   └── run_gaussian.py       # Runs BaM on a synthetic Gaussian target
└── results/
    ├── output.txt            # Console output (produced by reproduce.sh)
    └── baum_convergence.png  # KL convergence plot
```

## How to reproduce the results

```bash
# Make the script executable
chmod +x reproduce.sh

# Run the reproduction script
./reproduce.sh
```

The script will:

1. Install the required Python packages (`numpy`, `scipy`, `matplotlib`, `tqdm`).
2. Run the example script that:
   * Generates a random 10‑dimensional Gaussian target.
   * Runs BaM for 200 iterations with a batch size of 200 and a regularization parameter λ = 50.
   * Records forward (`KL(p||q)`) and reverse (`KL(q||p)`) KL divergences.
   * Saves the convergence plot to `results/baum_convergence.png`.
3. Capture all console output in `results/output.txt`.

## What the example produces

* `results/output.txt` – prints the KL values at each iteration and the final values.
* `results/baum_convergence.png` – a log‑scale plot showing how the forward and reverse KL divergences decrease over iterations.

The example demonstrates the exponential‑like convergence described in the paper (Section 3.2) for a Gaussian target.  The hyper‑parameters (batch size, λ, number of iterations) are chosen to match the values reported in the experiments of the paper.

Feel free to modify `examples/run_gaussian.py` to change dimensionality, batch size, or λ to explore different regimes.  The code is fully documented and should serve as a starting point for reproducing more extensive experiments from the paper.

---

**Note**: This repository contains only the core BaM implementation and a minimal reproducible experiment.  Full experimental suites (hierarchical models, deep generative models, baseline comparisons) are omitted for brevity, but the structure is ready for extension if desired.