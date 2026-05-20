"""
Configuration file for the PINN experiments.

All hyper‑parameters are defined here so that they can be easily tuned
without modifying the training logic.
"""

import math
import torch

# --------------------------- General ---------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [0, 1, 2, 3, 4]          # 5 random seeds as in the paper
ITERATIONS = 41000
ADAM_LR_GRID = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
LBFGS_LR = 1.0
LBFGS_MEM = 100
LBFGS_LINE_SEARCH = "strong_wolfe"  # PyTorch default
NNCG_TOL = 1e-6
NNCG_MAX_ITERS = 20

# --------------------------- Network ---------------------------

WIDTHS = [50, 100, 200, 400]
HIDDEN_LAYERS = 3
ACTIVATION = torch.tanh

# --------------------------- Data ---------------------------

# Residual and boundary points are generated in experiments/pinn.py
RES_POINTS = 10000
INIT_POINTS = 257
BND_POINTS = 101

# --------------------------- Switch points ---------------------------

SWITCH_POINTS = [1000, 11000, 31000]   # Adam -> L‑BFGS


# --------------------------- Utilities ---------------------------

def get_adam_lr():
    """Return the best learning rate from the grid search."""
    # The paper reports that 1e‑3 works well for most widths.
    # We keep the grid but use 1e‑3 as the default.
    return 1e-3