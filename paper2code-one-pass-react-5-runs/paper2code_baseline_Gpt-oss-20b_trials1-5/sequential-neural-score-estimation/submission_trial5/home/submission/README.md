# Sequential Neural Posterior Score Estimation (toy implementation)

This repository contains a minimal, fully‑reproducible implementation of the
**Sequential Neural Posterior Score Estimation (NPSE)** method described in
the paper *“Sequential Neural Score Estimation: Likelihood‑Free Inference with
Conditional Score Based Diffusion Models”*.

The implementation is deliberately small – it trains a diffusion‑based
inference model on a toy Gaussian linear simulator and produces posterior
samples for a single observation.  The code is fully self‑contained,
requires only `torch` and `scipy`, and can be executed on any Ubuntu 24.04
machine (or in the provided Docker image) with a single command:

```bash
bash reproduce.sh
```

The script will:

1.  Install the required Python packages.
2.  Train an NPSE model on 10 000 simulated data points.
3.  Generate 5 000 posterior samples for the observation `x_obs = [0.0, 0.0]`.
4.  Save the samples to `posterior_samples.csv`.

The repository is intentionally lightweight (≈ 0.6 MB) – only the source
code is committed, no heavy artefacts.

## Repository layout

```
/home/submission/
├── README.md
├── reproduce.sh
├── requirements.txt
└── src/
    ├── main.py
    ├── model.py
    └── utils.py
```

- `reproduce.sh` – the reproduction script.
- `requirements.txt` – Python dependencies.
- `src/` – source code for the toy NPSE implementation.

Feel free to modify the hyper‑parameters or extend the simulator for your
own experiments.

---

### Running locally

```bash
# Make sure the script is executable
chmod +x reproduce.sh

# Run the reproduction script
bash reproduce.sh
```

After it finishes you will find a file `posterior_samples.csv` in the
root directory containing the generated posterior samples.

### Expected outputs

The CSV file contains 5 000 rows and two columns (`θ₁,θ₂`).  The values
should be centred around the true parameter `θ_true = [0.0, 0.0]` with
variance roughly `0.1` (the posterior variance for this toy problem).

You can visualise the samples with any plotting tool, e.g.:

```bash
python - <<'PY'
import pandas as pd, matplotlib.pyplot as plt
df = pd.read_csv('posterior_samples.csv')
plt.scatter(df['θ₁'], df['θ₂'], alpha=0.3, s=5)
plt.xlabel('θ₁'); plt.ylabel('θ₂'); plt.title('Posterior samples');
plt.axis('equal'); plt.grid(True); plt.show()
PY
```

---

### Acknowledgements

The code is inspired by the original NPSE paper and the
`score‑based‑inference` library
(https://github.com/hojonathanho/pytorch-diffusion).  The design choices
here are intentionally simple to keep the repository lightweight while
still demonstrating the core ideas of conditional diffusion for
simulation‑based inference.