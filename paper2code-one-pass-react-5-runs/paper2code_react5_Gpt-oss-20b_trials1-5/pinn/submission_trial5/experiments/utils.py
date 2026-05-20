"""
Utility functions for metrics and evaluation.
"""

import numpy as np
import torch
from scipy.interpolate import griddata

def l2relative_error(pred: torch.Tensor, true: torch.Tensor) -> float:
    """L2 relative error over all evaluation points."""
    diff = pred - true
    return torch.sqrt(torch.sum(diff ** 2) / torch.sum(true ** 2)).item()


def evaluate_solution(model, pde, eval_points: np.ndarray) -> float:
    """Compute L2RE of the model over a dense evaluation grid."""
    # eval_points: (N, 2) with x and t
    with torch.no_grad():
        x = torch.tensor(eval_points[:, 0], device=model.device).float()
        t = torch.tensor(eval_points[:, 1], device=model.device).float()
        inp = torch.stack([x, t], dim=1)
        pred = model(inp).squeeze()
        true = pde.analytical(x, t)
    return l2relative_error(pred, true)


def create_evaluation_grid(pde):
    """Return a dense grid for evaluation (same as training grid)."""
    if isinstance(pde, (Convection, Reaction)):
        x = np.linspace(0, PI2, 255)
        t = np.linspace(0, 1, 100)
    else:  # Wave
        x = np.linspace(0, 1, 255)
        t = np.linspace(0, 1, 100)
    grid = _grid(x, t)
    return grid