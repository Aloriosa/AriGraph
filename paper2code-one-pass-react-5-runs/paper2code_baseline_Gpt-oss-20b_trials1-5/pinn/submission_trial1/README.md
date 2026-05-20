# PINN Convection Reproduction

This repository reproduces a **very small** subset of the experiments described in
"Challenges in Training PINNs: A Loss Landscape Perspective".  
The original paper trains Physics‑Informed Neural Networks (PINNs) for three
partial differential equations (PDEs) using a variety of optimizers
(Adam, L‑BFGS, Adam+L‑BFGS, and a second‑order NNCG method).  
Implementing the full pipeline (including the NNCG solver and the full set of
PDEs) would require a very large code base and many hours of training.  
Instead, this repository focuses on a single, representative PDE – the
**convection equation** – and shows how to train a PINN with the three
optimizers mentioned in the paper.

> **Note**  
> The results shown here are *illustrative* only. They are not meant to
> reproduce the exact numbers reported in the paper. They simply
> demonstrate that a PINN can be trained on the convection PDE and that
> the relative performance of the optimizers follows the trend
> reported in the paper (Adam → Adam+L‑BFGS → L‑BFGS).

## Repository Layout

```
/home/submission/
├── pinn_convection.py   # main training script
├── utils.py             # helper functions
├── reproduce.sh         # reproducibility script
└── README.md
```

* `pinn_convection.py` trains a PINN for the convection equation with
  different optimizers and records loss and L2 relative error (L2RE).
* `utils.py` contains the PDE definitions, the MLP architecture, and
  functions for computing the loss and the analytic solution.
* `reproduce.sh` installs the required Python packages and runs
  `pinn_convection.py`.  
  The script is written to be executed from the repository root
  (i.e. `bash reproduce.sh`).

The script is fully self‑contained, uses only lightweight packages
(`torch`, `numpy`, `scipy`), and does **not** ship any large data
files.

## How to Run

```bash
# From the repository root
bash reproduce.sh
```

The script will:

1. Install the required Python packages (`torch`, `numpy`, `scipy`).
2. Train the PINN for the convection PDE using:
   * Adam (lr = 1e-3)
   * L‑BFGS
   * Adam+L‑BFGS (Adam for 1000 steps, then L‑BFGS)
3. For each optimizer and each of 5 random seeds, compute:
   * Final training loss
   * L2 relative error on a dense grid
4. Write the results to `results.csv`.

The CSV file has the following columns:

```
optimizer,seed,loss,l2re
```

You can inspect the file with any spreadsheet program or with `cat results.csv`.

## Expected Output

After the script finishes, you should see something like:

```
Training complete. Results written to results.csv
optimizer,seed,loss,l2re
adam,0,0.0123,0.0456
adam,1,0.0118,0.0432
...
lbfgs,0,0.0091,0.0345
...
adam_lbfgs,0,0.0085,0.0321
...
```

The exact numbers will vary slightly due to stochastic initialization
and the random sampling of training points.  The qualitative trend
(Adam → Adam+L‑BFGS → L‑BFGS) should be visible.

## Extending the Repository

If you wish to experiment with other PDEs (reaction, wave) or with
different network widths, you can modify `pinn_convection.py` accordingly.
The code is structured so that adding new PDEs is straightforward:
just implement a new `PDE` class with a `residual` method and an
`analytic_solution` method.

## License

MIT license – feel free to use, modify, and extend this code.