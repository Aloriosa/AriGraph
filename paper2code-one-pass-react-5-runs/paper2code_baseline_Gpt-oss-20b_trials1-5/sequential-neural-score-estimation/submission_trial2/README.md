# Sequential Neural Score Estimation (SNPSE) – Reproduction Repository

This repository contains a minimal, fully reproducible implementation of the *Sequential Neural Posterior Score Estimation* (SNPSE) method introduced in  
> Louis Sharrock, Jack Simons, Song Liu, & Mark Beaumont.  
> **Sequential Neural Score Estimation: Likelihood‑Free Inference with Conditional Score Based Diffusion Models**  
> *ICML 2024*.

The code reproduces the toy experiments from the paper as well as the Gaussian‑Linear benchmark.  
All heavy simulation data are re‑generated on the fly, so the repository stays well below 1 GB.

## Quick start

```bash
# 1. Create a fresh environment
python -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
bash reproduce.sh

# 3. Run the toy experiment
python scripts/train_npse.py --benchmark toy --seed 42

# 4. Run the Gaussian‑Linear experiment
python scripts/train_npse.py --benchmark gaussian_linear --seed 42
python scripts/train_tsnpse.py --benchmark gaussian_linear --seed 42

# 5. Sample from the trained posterior
python scripts/sample.py --model npse_gaussian_linear_5000.pt --n_samples 5000
```

The script `reproduce.sh` installs the required packages and runs a short sanity‑check that trains NPSE on the toy benchmark and the Gaussian‑Linear benchmark.  
All code uses deterministic seeds so the results are fully reproducible.

## Repository layout

```
.
├── README.md
├── reproduce.sh
├── requirements.txt
├── src
│   ├── __init__.py
│   ├── dataset.py
│   ├── score_network.py
│   ├── utils.py
│   ├── npse.py
│   ├── tsnpse.py
│   ├── trainer_npse.py
│   └── trainer_tsnpse.py
└── scripts
    ├── train_npse.py
    ├── train_tsnpse.py
    └── sample.py
```

* **src/** – Core implementation (dataset generation, MLP score network, NPSE/TSNPSE training loops).
* **scripts/** – Simple command‑line wrappers to train and sample.
* **reproduce.sh** – Installs dependencies and runs a quick demo.
* **requirements.txt** – Python dependencies.

## How the code relates to the paper

| Paper section | Repository component | Notes |
|---------------|----------------------|-------|
| 2.2 – Diffusion model | `src/npse.py` | Implements the forward SDE, score‑matching loss, and probability‑flow ODE sampling. |
| 3.1 – TSNPSE | `src/tsnpse.py` | Adds truncated proposals and HPR estimation. |
| 5.1 – Benchmark details | `src/dataset.py` | Implements the Gaussian‑Linear and toy benchmarks. |
| 5.2 – Experimental details | `src/trainer_npse.py`, `src/trainer_tsnpse.py` | Fixed architecture (3×256 MLP, SiLU), Adam(1e‑4), early stopping on 15 % validation set. |
| 5.3 – Real‑world experiment | Not included (requires domain‑specific simulator). | The repository focuses on the toy and Gaussian‑Linear benchmarks which are fully self‑contained. |

The code follows the same algorithmic flow as in the paper:  
1. Generate simulated data `(θ₀, x)` from the prior and simulator.  
2. Perturb `θ₀` to `θ_t` via the forward SDE.  
3. Train a time‑dependent score network `s(θ_t, x, t)` by minimizing the denoising posterior score‑matching loss.  
4. Sample from the posterior by integrating the probability‑flow ODE from the reference distribution back to `t=0`.

Feel free to extend the repository to other benchmarks from the SBIBM library – the modular design makes this straightforward.

Enjoy!