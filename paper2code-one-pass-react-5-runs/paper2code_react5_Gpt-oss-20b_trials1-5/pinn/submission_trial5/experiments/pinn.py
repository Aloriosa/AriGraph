"""
PDE definitions and loss computation.

All PDEs are implemented in the same interface:
- `sample_points()` returns a dict with points for residual, initial and boundary.
- `residual()` computes the differential operator applied to the network output.
- `boundary()` enforces Dirichlet or Neumann conditions.
"""

import math
import torch
import numpy as np
from typing import Dict

# Common constants
PI2 = 2 * math.pi
PI = math.pi


def _grid(x1: np.ndarray, x2: np.ndarray):
    """Return a mesh grid flattened to (N, 2)."""
    X1, X2 = np.meshgrid(x1, x2, indexing="ij")
    return np.vstack([X1.ravel(), X2.ravel()]).T


class Convection:
    """One‑dimensional linear convection problem."""
    beta = 40.0

    @staticmethod
    def sample_points(n_res, init_pts, bnd_pts):
        # Uniform grid in [0,2π]×[0,1]
        x = np.linspace(0, PI2, 255, dtype=np.float32)
        t = np.linspace(0, 1, 100, dtype=np.float32)
        interior = _grid(x, t)[1:-1]  # drop boundaries
        res = interior[np.random.choice(len(interior), n_res, replace=False)]

        init = np.c_[np.full(init_pts, 0.0), np.linspace(0, 1, init_pts, dtype=np.float32)]
        bnd1 = np.c_[np.full(bnd_pts, 0.0), np.linspace(0, 1, bnd_pts, dtype=np.float32)]
        bnd2 = np.c_[np.full(bnd_pts, PI2), np.linspace(0, 1, bnd_pts, dtype=np.float32)]
        return {"res": res, "init": init, "bnd1": bnd1, "bnd2": bnd2}

    @staticmethod
    def analytical(x, t):
        return torch.sin(x - Convection.beta * t)

    @staticmethod
    def residual(u, x, t):
        # u_t + beta u_x
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), retain_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True)[0]
        return u_t + Convection.beta * u_x

    @staticmethod
    def boundary(u, x, t, val):
        return u - val


class Reaction:
    """One‑dimensional reaction ODE."""
    rho = 5.0

    @staticmethod
    def sample_points(n_res, init_pts, bnd_pts):
        x = np.linspace(0, PI2, 255, dtype=np.float32)
        t = np.linspace(0, 1, 100, dtype=np.float32)
        interior = _grid(x, t)[1:-1]
        res = interior[np.random.choice(len(interior), n_res, replace=False)]

        init = np.c_[np.full(init_pts, 0.0), np.linspace(0, 1, init_pts, dtype=np.float32)]
        bnd1 = np.c_[np.full(bnd_pts, 0.0), np.linspace(0, 1, bnd_pts, dtype=np.float32)]
        bnd2 = np.c_[np.full(bnd_pts, PI2), np.linspace(0, 1, bnd_pts, dtype=np.float32)]
        return {"res": res, "init": init, "bnd1": bnd1, "bnd2": bnd2}

    @staticmethod
    def analytical(x, t):
        h = torch.exp(-(x - PI) ** 2 / (2 * (PI / 4) ** 2))
        return h * torch.exp(Reaction.rho * t) / (h * torch.exp(Reaction.rho * t) + 1 - h)

    @staticmethod
    def residual(u, x, t):
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), retain_graph=True)[0]
        return u_t - Reaction.rho * u * (1 - u)

    @staticmethod
    def boundary(u, x, t, val):
        return u - val


class Wave:
    """One‑dimensional wave equation."""
    beta = 5.0

    @staticmethod
    def sample_points(n_res, init_pts, bnd_pts):
        x = np.linspace(0, 1, 255, dtype=np.float32)
        t = np.linspace(0, 1, 100, dtype=np.float32)
        interior = _grid(x, t)[1:-1]
        res = interior[np.random.choice(len(interior), n_res, replace=False)]

        # Initial condition at t=0
        init = np.c_[np.linspace(0, 1, init_pts, dtype=np.float32), np.zeros(init_pts, dtype=np.float32)]
        # Boundary conditions at x=0 and x=1
        bnd1 = np.c_[np.zeros(bnd_pts, dtype=np.float32), np.linspace(0, 1, bnd_pts, dtype=np.float32)]
        bnd2 = np.c_[np.ones(bnd_pts, dtype=np.float32), np.linspace(0, 1, bnd_pts, dtype=np.float32)]
        return {"res": res, "init": init, "bnd1": bnd1, "bnd2": bnd2}

    @staticmethod
    def analytical(x, t):
        return torch.sin(PI * x) * torch.cos(2 * PI * t) + 0.5 * torch.sin(Wave.beta * PI * x) * torch.cos(2 * Wave.beta * PI * t)

    @staticmethod
    def residual(u, x, t):
        u_tt = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), retain_graph=True)[0]
        u_tt = torch.autograd.grad(u_tt, t, grad_outputs=torch.ones_like(u), retain_graph=True)[0]
        u_xx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True)[0]
        u_xx = torch.autograd.grad(u_xx, x, grad_outputs=torch.ones_like(u), retain_graph=True)[0]
        return u_tt - 4 * u_xx

    @staticmethod
    def boundary(u, x, t, val):
        return u - val