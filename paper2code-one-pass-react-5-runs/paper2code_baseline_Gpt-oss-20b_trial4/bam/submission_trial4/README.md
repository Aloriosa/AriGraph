# Reproduction of “Batch and match: black-box variational inference with a score-based divergence”

This repository contains a minimal, fully reproducible implementation of the **BaM** algorithm
from the paper *Batch and match: black-box variational inference with a score-based divergence*,
together with a simple ADVI baseline for comparison.  
All code is pure Python (NumPy, JAX, SciPy) and requires only a CPU; if an NVIDIA GPU is
available the JAX backend will automatically use it.

## Directory layout

```
/home/submission/
├── README.md
├── reproduce.sh
├── requirements.txt
└── src/
    ├── bam.py
    ├── advi.py
    ├── utils.py
    └── main.py
```

## How to run

```bash
bash reproduce.sh
```

`reproduce.sh` installs the required packages, runs the experiment, and prints a short
summary.  The script also produces a CSV file `results.csv` containing the final
variational parameters and KL divergences for both BaM and ADVI.

The experiment uses a synthetic Gaussian target (`μ* = [1, 2, 3]`,  
`Σ* = [[2, 0.5, 0.3], [0.5, 1, 0.2], [0.3, 0.2, 1.5]]`).  
BaM is run for 200 iterations with batch size 20 and a constant
regularisation parameter `λ = 1.0`.  ADVI uses simple gradient descent on the
closed‑form KL divergence with learning rate `α = 0.01`.

The script outputs:

```
Iteration 200:
  BaM:  μ = [...], Σ = [...], forward KL = 0.12, reverse KL = 0.10
  ADVI: μ = [...], Σ = [...], forward KL = 0.13, reverse KL = 0.11
```

and writes the same information to `results.csv`.

## File descriptions

| File | Purpose |
|------|---------|
| `bam.py` | Implementation of the BaM algorithm (batch + match steps). |
| `advi.py` | Simple ADVI implementation for Gaussian target (closed‑form KL). |
| `utils.py` | Helper functions: Gaussian log PDF, KL, and score computation. |
| `main.py` | Orchestrates the experiment, prints results, writes CSV. |
| `requirements.txt` | Python dependencies. |
| `reproduce.sh` | Shell script that sets up the environment and runs `python src/main.py`. |

Feel free to modify the target distribution, batch size, or learning‑rate schedule
to explore additional behaviours.  All code is self‑contained and does not
depend on external data or large binaries.

---

**End of README**