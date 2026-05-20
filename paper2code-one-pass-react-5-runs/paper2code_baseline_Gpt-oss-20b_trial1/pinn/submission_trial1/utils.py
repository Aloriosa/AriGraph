"""
Utility functions and classes for PINN training on the convection PDE.
"""

import torch
import numpy as np

# --------------------------------------------------------------------------- #
# 1. PDE Definition
# --------------------------------------------------------------------------- #
class ConvectionPDE:
    """
    One‑dimensional convection PDE:
        ∂u/∂t + β ∂u/∂x = 0,
    with periodic boundary in x and initial condition u(x,0)=sin(x).
    The analytic solution is u(x,t)=sin(x-βt).
    """
    def __init__(self, beta=40.0, device="cpu"):
        self.beta = beta
        self.device = device

    def analytic_solution(self, x, t):
        """Return exact solution u(x,t)=sin(x-βt)."""
        return torch.sin(x - self.beta * t)

    def residual(self, model, points):
        """
        Compute PDE residual at given points.

        Parameters
        ----------
        model : torch.nn.Module
            The PINN.
        points : torch.Tensor
            Tensor of shape (N,2) with columns (x,t).

        Returns
        -------
        res : torch.Tensor
            Residual values of shape (N,).
        """
        x = points[:, 0].unsqueeze(1)  # (N,1)
        t = points[:, 1].unsqueeze(1)  # (N,1)
        inputs = torch.cat([x, t], dim=1)  # (N,2)

        inputs.requires_grad_(True)
        u = model(inputs).squeeze()  # (N,)

        grads = torch.autograd.grad(
            u,
            inputs,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True,
        )[0]  # (N,2)

        u_t = grads[:, 1]
        u_x = grads[:, 0]

        res = u_t + self.beta * u_x
        return res

    def boundary_loss(self, model, points, target):
        """
        Compute boundary condition loss |u(x,t)-target|^2.

        Parameters
        ----------
        model : torch.nn.Module
            The PINN.
        points : torch.Tensor
            (N,2) tensor of (x,t).
        target : torch.Tensor
            (N,) tensor of target values.

        Returns
        -------
        loss : torch.Tensor
            Mean squared error over the points.
        """
        u = model(points).squeeze()
        return torch.mean((u - target) ** 2)

# --------------------------------------------------------------------------- #
# 2. MLP Architecture
# --------------------------------------------------------------------------- #
class PINNMLP(torch.nn.Module):
    """
    Three‑layer MLP with tanh activations.

    Arguments
    ---------
    width : int
        Number of neurons per hidden layer.
    """
    def __init__(self, width=50):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2, width),
            torch.nn.Tanh(),
            torch.nn.Linear(width, width),
            torch.nn.Tanh(),
            torch.nn.Linear(width, width),
            torch.nn.Tanh(),
            torch.nn.Linear(width, 1),
        )
        # Xavier normal initialization
        for m in self.net:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

# --------------------------------------------------------------------------- #
# 3. Data Sampling
# --------------------------------------------------------------------------- #
def sample_residual_points(n_res, device="cpu", seed=None):
    """Sample residual points uniformly in the domain (x∈[0,2π], t∈[0,1])."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 2 * np.pi, size=n_res)
    t = rng.uniform(0, 1, size=n_res)
    points = np.stack([x, t], axis=1).astype(np.float32)
    return torch.tensor(points, device=device)

def sample_initial_points(n_init, device="cpu", seed=None):
    """Sample initial condition points (t=0)."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 2 * np.pi, size=n_init)
    t = np.zeros_like(x)
    points = np.stack([x, t], axis=1).astype(np.float32)
    return torch.tensor(points, device=device)

def sample_boundary_points(n_bdy, device="cpu", seed=None):
    """Sample boundary points at x=0 and x=2π."""
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, 1, size=n_bdy)
    x_left = np.zeros_like(t)
    x_right = np.full_like(t, 2 * np.pi)
    points_left = np.stack([x_left, t], axis=1).astype(np.float32)
    points_right = np.stack([x_right, t], axis=1).astype(np.float32)
    return torch.tensor(points_left, device=device), torch.tensor(points_right, device=device)

# --------------------------------------------------------------------------- #
# 4. Grid for L2 Relative Error
# --------------------------------------------------------------------------- #
def create_evaluation_grid(nx=255, nt=100, device="cpu"):
    """Create a dense grid for evaluating the solution."""
    x = torch.linspace(0, 2 * np.pi, steps=nx, device=device)
    t = torch.linspace(0, 1, steps=nt, device=device)
    X, T = torch.meshgrid(x, t, indexing="ij")
    points = torch.stack([X.flatten(), T.flatten()], dim=1)
    return points

# --------------------------------------------------------------------------- #
# 5. Training Utilities
# --------------------------------------------------------------------------- #
def l2re(pred, true):
    """Compute L2 relative error."""
    num = torch.mean((pred - true) ** 2)
    den = torch.mean(true ** 2)
    return torch.sqrt(num / den).item()