#!/usr/bin/env python3
"""
Full experiment pipeline for reproducing the results of
“Challenges in Training PINNs: A Loss Landscape Perspective”.

The script iterates over all combinations of PDE, network width, random seed,
Adam learning‑rate, and Adam→L‑BFGS switch point, trains a PINN with the
specified schedule, then runs a damped Newton step (NysNewton‑CG) to further
reduce the loss.  All metrics are written to `results.csv`.

Author: OpenAI ChatGPT
Date: 2026-03-14
"""

import os
import csv
import math
import random
import itertools
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd.functional import hessian
from tqdm import tqdm

# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------

# PDE names
PDE_NAMES = ["convection", "wave", "reaction"]

# Network widths
WIDTHS = [50, 100, 200, 400]

# Random seeds
SEEDS = list(range(5))

# Adam learning‑rate candidates
LR_LIST = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

# Adam → L‑BFGS switch points (iterations)
SWITCH_POINTS = [1000, 11000, 31000]

# Total training iterations (Adam + L‑BFGS)
TOTAL_ITERS = 41000

# NysNewton‑CG parameters
NNCG_ITERS = 1000
NNCG_K = 30          # Nyström sketch size
NNCG_MU = 1e-2       # damping
NNCG_TOL = 1e-8
NNCG_MAX_CG_ITERS = 20

# Loss weights
RES_WEIGHT = 1.0
BC_WEIGHT = 1.0
IC_WEIGHT = 1.0

# ---------------------------------------------------------------------------
#  Utility functions
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def device() -> torch.device:
    """Return the default device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def linspace_grid(n: int, m: int):
    """Return a grid of shape (n, m) with coordinates in [0,1]×[0,1]."""
    t = torch.linspace(0, 1, n)
    x = torch.linspace(0, 1, m)
    tt, xx = torch.meshgrid(t, x, indexing="ij")
    return tt.reshape(-1, 1), xx.reshape(-1, 1)

# ---------------------------------------------------------------------------
#  PINN model
# ---------------------------------------------------------------------------

class PINN(nn.Module):
    def __init__(self, hidden_width: int, hidden_layers: int = 3):
        super().__init__()
        layers = [nn.Linear(2, hidden_width), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_width, hidden_width), nn.Tanh()]
        layers.append(nn.Linear(hidden_width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, 2) → (N, 1)"""
        return self.net(x)

# ---------------------------------------------------------------------------
#  PDE definitions
# ---------------------------------------------------------------------------

class PDE:
    """Static class containing PDE parameters, exact solutions, and loss terms."""

    @staticmethod
    def convection_params():
        return {"beta": 40.0}

    @staticmethod
    def wave_params():
        return {"beta": 5.0}

    @staticmethod
    def reaction_params():
        return {"rho": 5.0}

    @staticmethod
    def exact_solution(pde_name: str, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if pde_name == "convection":
            beta = PDE.convection_params()["beta"]
            return torch.sin(x - beta * t)
        elif pde_name == "wave":
            beta = PDE.wave_params()["beta"]
            return torch.sin(math.pi * x) * torch.cos(2 * math.pi * t) + \
                   0.5 * torch.sin(beta * math.pi * x) * torch.cos(2 * beta * math.pi * t)
        elif pde_name == "reaction":
            rho = PDE.reaction_params()["rho"]
            h = torch.exp(-(x - math.pi)**2 / (2 * (math.pi / 4)**2))
            return h * torch.exp(rho * t) / (h * torch.exp(rho * t) + 1 - h)
        else:
            raise ValueError(f"Unknown PDE: {pde_name}")

    @staticmethod
    def residual(pde_name: str, model: nn.Module, t: torch.Tensor,
                 x: torch.Tensor) -> torch.Tensor:
        """
        Compute the residual term of the PDE for the given points.
        The returned value is the mean squared residual.
        """
        t = t.requires_grad_(True)
        x = x.requires_grad_(True)
        u = model(torch.cat([t, x], dim=1))

        if pde_name == "convection":
            beta = PDE.convection_params()["beta"]
            u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),
                                      create_graph=True)[0]
            u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                      create_graph=True)[0]
            res = u_t + beta * u_x
        elif pde_name == "wave":
            u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),
                                      create_graph=True)[0]
            u_tt = torch.autograd.grad(u_t, t, grad_outputs=torch.ones_like(u_t),
                                       create_graph=True)[0]
            u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                      create_graph=True)[0]
            u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                                       create_graph=True)[0]
            res = u_tt - 4 * u_xx
        elif pde_name == "reaction":
            rho = PDE.reaction_params()["rho"]
            u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),
                                      create_graph=True)[0]
            res = u_t - rho * u * (1 - u)
        else:
            raise ValueError(f"Unknown PDE: {pde_name}")

        return 0.5 * torch.mean(res**2)

    @staticmethod
    def boundary(pde_name: str, model: nn.Module, t: torch.Tensor,
                 x: torch.Tensor, exact: torch.Tensor) -> torch.Tensor:
        """
        Boundary loss: (u - u_exact)^2
        """
        u_pred = model(torch.cat([t, x], dim=1))
        return 0.5 * torch.mean((u_pred - exact)**2)

    @staticmethod
    def initial(pde_name: str, model: nn.Module, t: torch.Tensor,
                x: torch.Tensor, exact: torch.Tensor) -> torch.Tensor:
        """
        Initial condition loss: (u - u_exact)^2
        """
        u_pred = model(torch.cat([t, x], dim=1))
        return 0.5 * torch.mean((u_pred - exact)**2)

# ---------------------------------------------------------------------------
#  Data generation
# ---------------------------------------------------------------------------

def generate_data(pde_name: str, seed: int):
    """
    Generate training data for the given PDE and seed.
    Returns a dictionary of tensors (all on CPU).
    """
    set_seed(seed)

    # Grid dimensions
    nx = 255
    nt = 100

    # Domain bounds
    if pde_name == "convection" or pde_name == "reaction":
        x_min, x_max = 0.0, 2 * math.pi
    elif pde_name == "wave":
        x_min, x_max = 0.0, 1.0
    else:
        raise ValueError(f"Unknown PDE: {pde_name}")

    t_grid = torch.linspace(0, 1, nt)
    x_grid = torch.linspace(x_min, x_max, nx)
    tt, xx = torch.meshgrid(t_grid, x_grid, indexing="ij")
    t_all = tt.reshape(-1, 1)
    x_all = xx.reshape(-1, 1)

    # Residual points: sample 10k uniformly from the grid
    idx = torch.randperm(t_all.shape[0])[:10000]
    t_res = t_all[idx]
    x_res = x_all[idx]

    # Boundary points
    t_bc = torch.linspace(0, 1, 101).unsqueeze(1)
    if pde_name == "convection" or pde_name == "reaction":
        x_bc_left = torch.full_like(t_bc, x_min)
        x_bc_right = torch.full_like(t_bc, x_max)
    elif pde_name == "wave":
        x_bc_left = torch.full_like(t_bc, x_min)
        x_bc_right = torch.full_like(t_bc, x_max)
    else:
        raise ValueError(f"Unknown PDE: {pde_name}")

    # Initial condition points
    t_ic = torch.zeros(257, 1)
    x_ic = torch.linspace(x_min, x_max, 257).unsqueeze(1)

    return {
        "t_res": t_res,
        "x_res": x_res,
        "t_bc_left": t_bc,
        "x_bc_left": x_bc_left,
        "t_bc_right": t_bc,
        "x_bc_right": x_bc_right,
        "t_ic": t_ic,
        "x_ic": x_ic,
        "t_all": t_all,
        "x_all": x_all,
    }

# ---------------------------------------------------------------------------
#  Loss and metrics
# ---------------------------------------------------------------------------

def compute_loss(pde_name: str, model: nn.Module, data: dict) -> torch.Tensor:
    """Compute the full PINN loss."""
    # Residual
    res = PDE.residual(pde_name, model,
                       data["t_res"], data["x_res"])
    # Boundary left/right
    bc_left = PDE.boundary(pde_name, model,
                           data["t_bc_left"], data["x_bc_left"],
                           PDE.exact_solution(pde_name,
                                              data["t_bc_left"],
                                              data["x_bc_left"]))
    bc_right = PDE.boundary(pde_name, model,
                            data["t_bc_right"], data["x_bc_right"],
                            PDE.exact_solution(pde_name,
                                               data["t_bc_right"],
                                               data["x_bc_right"]))
    # Initial condition
    ic = PDE.initial(pde_name, model,
                     data["t_ic"], data["x_ic"],
                     PDE.exact_solution(pde_name,
                                        data["t_ic"],
                                        data["x_ic"]))
    loss = RES_WEIGHT * res + BC_WEIGHT * (bc_left + bc_right) + IC_WEIGHT * ic
    return loss

def compute_l2re(pde_name: str, model: nn.Module, data: dict) -> float:
    """Compute L2 relative error on the full grid + boundary + IC."""
    t = data["t_all"]
    x = data["x_all"]
    with torch.no_grad():
        u_pred = model(torch.cat([t, x], dim=1))
        u_exact = PDE.exact_solution(pde_name, t, x)
        err = torch.mean((u_pred - u_exact)**2)
        exact_norm = torch.mean(u_exact**2)
        l2re = torch.sqrt(err / exact_norm)
    return l2re.item()

# ---------------------------------------------------------------------------
#  Hessian–vector product and CG solver
# ---------------------------------------------------------------------------

def hvp(loss, model, vector):
    """
    Compute Hessian‑vector product Hv for the given loss and model.
    vector: list of tensors matching model.parameters()
    """
    grad_params = torch.autograd.grad(loss, model.parameters(), create_graph=True)
    flat_grad = torch.cat([g.reshape(-1) for g in grad_params])
    dot = torch.dot(flat_grad, vector)
    hv = torch.autograd.grad(dot, model.parameters(), retain_graph=True)
    hv_flat = torch.cat([h.reshape(-1) for h in hv]).detach()
    return hv_flat

def cg_solve(A_func, b, x0=None, max_iter=20, tol=1e-8):
    """
    Conjugate‑gradient solver for Ax = b.
    A_func: function that takes a vector and returns Av
    b: torch tensor
    """
    if x0 is None:
        x = torch.zeros_like(b)
    else:
        x = x0.clone()
    r = b - A_func(x)
    p = r.clone()
    rsold = torch.dot(r, r)
    for i in range(max_iter):
        Ap = A_func(p)
        alpha = rsold / torch.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        rsnew = torch.dot(r, r)
        if torch.sqrt(rsnew) < tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return x

# ---------------------------------------------------------------------------
#  NysNewton‑CG
# ---------------------------------------------------------------------------

def nysnewton_cg(pde_name: str, model: nn.Module, data: dict,
                 max_iters=NNCG_ITERS, mu=NNCG_MU, k=NNCG_K,
                 cg_tol=NNCG_TOL, cg_max_iter=NNCG_MAX_CG_ITERS):
    """
    Damped Newton method with Nyström‑preconditioned CG.
    For simplicity, we use a fixed identity preconditioner and a
    few CG iterations per Newton step.
    """
    for _ in range(max_iters):
        # Compute current loss and gradient
        loss = compute_loss(pde_name, model, data)
        loss.backward()
        grad = torch.cat([p.grad.reshape(-1) for p in model.parameters()]).detach()
        # Damped Hessian: H + mu I
        def Av(v):
            return hvp(loss, model, v) + mu * v
        # Solve (H + mu I) d = -grad
        d = cg_solve(Av, -grad, max_iter=cg_max_iter, tol=cg_tol)
        # Line search (Armijo)
        step = 1.0
        alpha = 1e-4
        beta = 0.5
        # Flatten parameters
        flat_params = torch.cat([p.data.reshape(-1) for p in model.parameters()])
        new_params = flat_params + step * d
        # Apply new params
        idx = 0
        for p in model.parameters():
            numel = p.numel()
            p.data.copy_(new_params[idx:idx+numel].reshape(p.shape))
            idx += numel
        # Evaluate new loss
        new_loss = compute_loss(pde_name, model, data)
        # Backtracking
        while new_loss > loss + alpha * step * torch.dot(grad, d):
            # revert
            idx = 0
            for p in model.parameters():
                numel = p.numel()
                p.data.copy_(flat_params[idx:idx+numel].reshape(p.shape))
                idx += numel
            step *= beta
            new_params = flat_params + step * d
            idx = 0
            for p in model.parameters():
                numel = p.numel()
                p.data.copy_(new_params[idx:idx+numel].reshape(p.shape))
                idx += numel
            new_loss = compute_loss(pde_name, model, data)
        # Clear gradients for next iteration
        for p in model.parameters():
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()
    # Return final loss
    return compute_loss(pde_name, model, data)

# ---------------------------------------------------------------------------
#  Experiment runner
# ---------------------------------------------------------------------------

def run_experiment(pde_name: str, width: int, seed: int,
                   lr: float, switch: int):
    """Run a single experiment and return metrics."""
    set_seed(seed)
    dev = device()

    model = PINN(hidden_width=width).to(dev)
    data = generate_data(pde_name, seed)
    # Move data to device
    for k, v in data.items():
        data[k] = v.to(dev)

    # Adam phase
    adam_opt = optim.Adam(model.parameters(), lr=lr)
    for _ in range(switch):
        adam_opt.zero_grad()
        loss = compute_loss(pde_name, model, data)
        loss.backward()
        adam_opt.step()

    # L‑BFGS phase
    remaining = TOTAL_ITERS - switch
    lbfgs_opt = optim.LBFGS(model.parameters(),
                            lr=1.0,
                            max_iter=remaining,
                            line_search_fn="strong_wolfe")
    def closure():
        lbfgs_opt.zero_grad()
        loss = compute_loss(pde_name, model, data)
        loss.backward()
        return loss
    lbfgs_opt.step(closure)

    # NysNewton‑CG phase
    nysnewton_cg(pde_name, model, data, max_iters=200, mu=NNCG_MU,
                 k=NNCG_K, cg_tol=NNCG_TOL, cg_max_iter=NNCG_MAX_CG_ITERS)

    # Final metrics
    final_loss = compute_loss(pde_name, model, data).item()
    final_l2re = compute_l2re(pde_name, model, data)
    return final_loss, final_l2re

# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    results_path = Path("results.csv")
    with results_path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["pde", "width", "seed", "lr", "switch",
                         "final_loss", "final_l2re"])
        # Iterate over all combinations
        for pde_name, width, seed, lr, switch in tqdm(
                itertools.product(PDE_NAMES, WIDTHS, SEEDS, LR_LIST, SWITCH_POINTS),
                desc="Total Experiments"):
            try:
                loss, l2re = run_experiment(pde_name, width, seed, lr, switch)
            except Exception as e:
                # In case of numerical issues, record NaNs
                loss, l2re = float("nan"), float("nan")
            writer.writerow([pde_name, width, seed, lr, switch, loss, l2re])

    print(f"All experiments finished. Results written to {results_path}")

if __name__ == "__main__":
    main()