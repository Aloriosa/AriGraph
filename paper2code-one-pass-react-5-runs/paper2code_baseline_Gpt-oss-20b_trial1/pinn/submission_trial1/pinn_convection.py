#!/usr/bin/env python3
"""
Train a PINN on the convection PDE using different optimizers
and report loss and L2 relative error.
"""

import torch
import numpy as np
import csv
import os
import time
import random
import argparse

from utils import (
    ConvectionPDE,
    PINNMLP,
    sample_residual_points,
    sample_initial_points,
    sample_boundary_points,
    create_evaluation_grid,
    l2re,
)

# --------------------------------------------------------------------------- #
# 1. Configuration
# --------------------------------------------------------------------------- #
NUM_RES_POINTS = 10000
NUM_INIT_POINTS = 257
NUM_BDY_POINTS = 101

MAX_ITER = 41000
ADAM_LR = 1e-3
LBFGS_HISTORY = 20
ADAM_BFGS_SWITCH = 1000  # Adam steps before switching to L-BFGS

SEEDS = [0, 1, 2, 3, 4]
WIDTH = 50  # one representative width

# --------------------------------------------------------------------------- #
# 2. Helper: Train with Adam
# --------------------------------------------------------------------------- #
def train_adam(model, pde, optim, closure, max_iter, device):
    best_loss = float("inf")
    for step in range(1, max_iter + 1):
        loss = optim.step(closure)
        if loss.item() < best_loss:
            best_loss = loss.item()
        if step % 5000 == 0:
            print(f"  Adam step {step}/{max_iter} loss={loss.item():.6f}")
    return best_loss

# --------------------------------------------------------------------------- #
# 3. Helper: Train with L-BFGS
# --------------------------------------------------------------------------- #
def train_lbfgs(model, pde, optim, closure, max_iter, device):
    best_loss = float("inf")
    for step in range(1, max_iter + 1):
        loss = optim.step(closure)
        if loss.item() < best_loss:
            best_loss = loss.item()
        if step % 5000 == 0:
            print(f"  L-BFGS step {step}/{max_iter} loss={loss.item():.6f}")
    return best_loss

# --------------------------------------------------------------------------- #
# 4. Main training loop
# --------------------------------------------------------------------------- #
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "results.csv")

    # Header for CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["optimizer", "seed", "loss", "l2re"])

    # Loop over seeds
    for seed in SEEDS:
        print(f"\n=== Seed {seed} ===")
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Sample data once (deterministic per seed)
        res_pts = sample_residual_points(NUM_RES_POINTS, device=device, seed=seed)
        init_pts = sample_initial_points(NUM_INIT_POINTS, device=device, seed=seed)
        bdy_left, bdy_right = sample_boundary_points(NUM_BDY_POINTS, device=device, seed=seed)

        # Initial condition target
        init_target = torch.sin(init_pts[:, 0]).to(device)
        # Periodic boundary target (u(0)=u(2π))
        left_val = torch.sin(bdy_left[:, 0]).to(device)  # sin(0)=0
        right_val = torch.sin(bdy_right[:, 0]).to(device)  # sin(2π)=0
        # For periodicity we enforce u(0)-u(2π)=0
        # We'll use a simple difference loss: (u_left - u_right)^2
        # (the target values are both zero, so same as u_left^2 + u_right^2)

        # Evaluation grid for L2RE
        eval_grid = create_evaluation_grid(device=device)

        # For each optimizer
        for optimizer_name in ["adam", "lbfgs", "adam_lbfgs"]:
            print(f"\n--- Optimizer: {optimizer_name} ---")
            model = PINNMLP(width=WIDTH).to(device)
            pde = ConvectionPDE(beta=40.0, device=device)

            # Loss closure
            def closure():
                model.zero_grad()
                # Residual loss
                res = pde.residual(model, res_pts)
                loss_res = torch.mean(res ** 2)

                # Initial condition loss
                ic = model(init_pts).squeeze()
                loss_ic = torch.mean((ic - init_target) ** 2)

                # Periodic boundary loss
                b_left = model(bdy_left).squeeze()
                b_right = model(bdy_right).squeeze()
                loss_bdy = torch.mean((b_left - b_right) ** 2)

                loss = loss_res + loss_ic + loss_bdy
                loss.backward()
                return loss

            # Choose optimizer
            if optimizer_name == "adam":
                optim = torch.optim.Adam(model.parameters(), lr=ADAM_LR)
                final_loss = train_adam(model, pde, optim, closure, MAX_ITER, device)
            elif optimizer_name == "lbfgs":
                optim = torch.optim.LBFGS(
                    model.parameters(),
                    lr=1.0,
                    max_iter=20,
                    max_eval=None,
                    tolerance_grad=1e-5,
                    tolerance_change=1e-9,
                    history_size=LBFGS_HISTORY,
                )
                final_loss = train_lbfgs(model, pde, optim, closure, MAX_ITER, device)
            else:  # adam_lbfgs
                # Adam phase
                optim_adam = torch.optim.Adam(model.parameters(), lr=ADAM_LR)
                print(f"  Adam phase for {ADAM_BFGS_SWITCH} steps")
                for step in range(1, ADAM_BFGS_SWITCH + 1):
                    loss = optim_adam.step(closure)
                    if step % 5000 == 0:
                        print(f"    Adam step {step}/{ADAM_BFGS_SWITCH} loss={loss.item():.6f}")
                # L-BFGS phase
                optim_lbfgs = torch.optim.LBFGS(
                    model.parameters(),
                    lr=1.0,
                    max_iter=20,
                    max_eval=None,
                    tolerance_grad=1e-5,
                    tolerance_change=1e-9,
                    history_size=LBFGS_HISTORY,
                )
                final_loss = train_lbfgs(model, pde, optim_lbfgs, closure, MAX_ITER - ADAM_BFGS_SWITCH, device)

            # Evaluate L2RE on dense grid
            with torch.no_grad():
                preds = model(eval_grid).squeeze()
                exact = pde.analytic_solution(eval_grid[:, 0], eval_grid[:, 1])
                error = l2re(preds, exact)

            print(f"  Final loss: {final_loss:.6f}   L2RE: {error:.6f}")

            # Save to CSV
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([optimizer_name, seed, f"{final_loss:.6e}", f"{error:.6e}"])

    print("\nTraining complete. Results written to", csv_path)


if __name__ == "__main__":
    main()