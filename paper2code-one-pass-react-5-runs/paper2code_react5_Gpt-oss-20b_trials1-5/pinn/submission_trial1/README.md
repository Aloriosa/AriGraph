# PINN Training – Reproduction of  
**“Challenges in Training PINNs: A Loss Landscape Perspective”**

This repository contains a fully reproducible implementation of the experimental
pipeline described in the paper.  The experiments are run on a single NVIDIA
Titan V GPU (CUDA 11.8) but fall back to CPU if no GPU is available.

## Highlights

* **Three benchmark PDEs** – *convection*, *wave*, *reaction*  
* **Four network widths** – 50, 100, 200, 400  
* **Five random seeds** – 0 … 4  
* **Adam learning‑rate grid** – 1e‑5, 1e‑4, 1e‑3, 1e‑2, 1e‑1  
* **Adam → L‑BFGS switch points** – 1 k, 11 k, 31 k  
* **Total training iterations** – 41 000 (Adam + L‑BFGS)  
* **NysNewton‑CG** – a damped Newton method with a Nyström‑preconditioned
  conjugate‑gradient solver is run after L‑BFGS to further reduce the loss.  
* **Evaluation** – L²‑relative error (L2RE) is computed on the full
  `255 × 100` interior grid plus the boundary/initial points.  
* **Results** – The script writes a single `results.csv` file with the
  following columns:  
  `pde,width,seed,lr,switch,final_loss,final_l2re`.  

All heavy artefacts are generated on the fly, so the repository size stays
well below 1 GB.  No pre‑trained checkpoints or large data files are stored.

## How to reproduce

```bash
bash reproduce.sh
```

The script will install the required Python packages (PyTorch 2.0, NumPy,
SciPy, tqdm) and run the full experiment pipeline.  After completion you
will find `results.csv` in the repository root.

> **Note**  
> The full set of experiments (4 widths × 5 seeds × 5 lr × 3 switches × 3 PDEs)
> would take several hours on a single GPU.  For quick sanity checks the script
> can be limited to a single combination by setting the environment variables
> `PDE`, `WIDTH`, `SEED`, `LR`, `SWITCH` before invoking `python src/train_pinn.py`.