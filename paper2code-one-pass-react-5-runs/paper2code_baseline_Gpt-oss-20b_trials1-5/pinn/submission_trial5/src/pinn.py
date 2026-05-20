#!/usr/bin/env python3
"""
Minimal Physics‑Informed Neural Network (PINN) implementation for the 1‑D wave
equation used in the paper “Challenges in Training PINNs: A Loss Landscape
Perspective”.

The network solves
    u_tt - 4 u_xx = 0   on (0,1)×(0,1)
with
    u(0,t)=u(1,t)=0,
    u(x,0)=sin(πx)+0.5 sin(βπx),
    u_t(x,0)=0  (β=5).

The loss is a weighted sum of the PDE residual, boundary conditions and
initial conditions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List


class MLP(nn.Module):
    """
    Three hidden layer MLP with tanh activations. The output is a scalar.
    """
    def __init__(self, width: int = 50):
        super().__init__()
        self.width = width
        self.fc1 = nn.Linear(2, width)
        self.fc2 = nn.Linear(width, width)
        self.fc3 = nn.Linear(width, width)
        self.out = nn.Linear(width, 1)

        # Xavier normal init
        for m in [self.fc1, self.fc2, self.fc3, self.out]:
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: x has shape (N,2) where columns are [x,t].
        Returns shape (N,1).
        """
        h = F.tanh(self.fc1(x))
        h = F.tanh(self.fc2(h))
        h = F.tanh(self.fc3(h))
        return self.out(h)


def _grad(y: torch.Tensor, x: torch.Tensor, create_graph: bool = True) -> torch.Tensor:
    """
    Compute gradient of y w.r.t. x. y shape (N,1), x shape (N,2).
    Returns y_grad shape (N,2).
    """
    grad = torch.autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=torch.ones_like(y),
        create_graph=create_graph,
        retain_graph=create_graph,
        only_inputs=True,
    )[0]
    return grad


def _second_derivative(u: torch.Tensor, x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Second derivative of u w.r.t. x[:,dim] (dim=0 for x, dim=1 for t).
    """
    du = _grad(u, x, create_graph=True)
    d2u = _grad(du[:, dim:dim+1], x, create_graph=True)
    return d2u


def pde_residual(u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Computes u_tt - 4 u_xx.
    """
    u_tt = _second_derivative(u, x, dim=1)
    u_xx = _second_derivative(u, x, dim=0)
    return u_tt - 4 * u_xx


def bc_loss(model: nn.Module, t_bc: torch.Tensor) -> torch.Tensor:
    """
    Boundary condition loss: u(0,t)=0, u(1,t)=0.
    t_bc: shape (N,1)
    """
    # left boundary
    x_left = torch.zeros_like(t_bc)
    inp_left = torch.cat([x_left, t_bc], dim=1)
    u_left = model(inp_left)
    loss_left = u_left.pow(2)

    # right boundary
    x_right = torch.ones_like(t_bc)
    inp_right = torch.cat([x_right, t_bc], dim=1)
    u_right = model(inp_right)
    loss_right = u_right.pow(2)

    return loss_left.mean() + loss_right.mean()


def ic_loss(model: nn.Module, x_ic: torch.Tensor) -> torch.Tensor:
    """
    Initial condition loss: u(x,0)=f(x), u_t(x,0)=0.
    """
    t0 = torch.zeros_like(x_ic)
    inp0 = torch.cat([x_ic, t0], dim=1)
    u0 = model(inp0)

    # target f(x) = sin(πx)+0.5 sin(βπx)
    beta = 5.0
    f = torch.sin(torch.pi * x_ic) + 0.5 * torch.sin(beta * torch.pi * x_ic)

    loss_u = (u0 - f).pow(2).mean()

    # u_t(x,0)
    u_t0 = _grad(u0, inp0, create_graph=True)[:, 1:2]
    loss_ut = u_t0.pow(2).mean()

    return loss_u + loss_ut


def loss_fn(
    model: nn.Module,
    x_res: torch.Tensor,
    x_bc: torch.Tensor,
    x_ic: torch.Tensor,
    weight_pde: float = 1.0,
    weight_bc: float = 1.0,
    weight_ic: float = 1.0,
) -> torch.Tensor:
    """
    Full PINN loss: weighted sum of PDE residual, boundary and initial conditions.
    """
    # PDE residual loss
    u = model(x_res)
    res = pde_residual(u, x_res)
    loss_pde = res.pow(2).mean()

    # Boundary loss
    loss_bc = bc_loss(model, x_bc)

    # Initial condition loss
    loss_ic = ic_loss(model, x_ic)

    return (
        weight_pde * loss_pde
        + weight_bc * loss_bc
        + weight_ic * loss_ic
    )


def l2re(
    model: nn.Module,
    grid_x: torch.Tensor,
    grid_t: torch.Tensor,
    device: torch.device,
) -> float:
    """
    Compute L2 relative error on the full 255×100 interior grid.
    """
    # Build full grid
    X, T = torch.meshgrid(grid_x, grid_t, indexing="ij")
    inp = torch.stack([X.flatten(), T.flatten()], dim=1).to(device)
    with torch.no_grad():
        u_pred = model(inp).squeeze().cpu().numpy()

    # Ground truth
    beta = 5.0
    u_gt = (
        torch.sin(torch.pi * X).cos() * torch.cos(2 * torch.pi * T)
        + 0.5 * torch.sin(beta * torch.pi * X).cos() * torch.cos(2 * beta * torch.pi * T)
    )
    u_gt = u_gt.reshape(-1).cpu().numpy()

    return np.linalg.norm(u_pred - u_gt) / np.linalg.norm(u_gt)


# --------------------------------------------------------------------------- #
# Helper to build data points
# --------------------------------------------------------------------------- #
import numpy as np


def build_points(
    n_res: int = 10000,
    n_bc: int = 101,
    n_ic: int = 257,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns: (x_res, x_bc, x_ic) all of shape (N,2) for PDE residual,
    (N_bc,1) for boundary t, (N_ic,1) for initial x.
    """
    # Residual points: uniform grid 255×100
    xs = np.linspace(0, 1, 255)
    ts = np.linspace(0, 1, 100)
    X, T = np.meshgrid(xs, ts, indexing="ij")
    x_res = np.stack([X.flatten(), T.flatten()], axis=1)
    x_res = torch.tensor(x_res, dtype=torch.float32, device=device)

    # Boundary points: t on [0,1] at x=0 and x=1
    t_bc = np.linspace(0, 1, n_bc).reshape(-1, 1)
    x_bc = torch.tensor(t_bc, dtype=torch.float32, device=device)

    # Initial points: x on [0,1] at t=0
    x_ic = np.linspace(0, 1, n_ic).reshape(-1, 1)
    x_ic = torch.tensor(x_ic, dtype=torch.float32, device=device)

    return x_res, x_bc, x_ic