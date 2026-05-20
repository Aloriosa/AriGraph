import math
import numpy as np
import torch

# Simple VESDE noise schedule (linear between sigma_min and sigma_max)
SIGMA_MIN = 0.0001
SIGMA_MAX = 15.0

def sigma_t(t: torch.Tensor) -> torch.Tensor:
    """
    Compute sigma(t) for the VESDE forward process.
    t is in [0,1].
    """
    return SIGMA_MIN + t * (SIGMA_MAX - SIGMA_MIN)

def mu_t(t: torch.Tensor) -> torch.Tensor:
    """
    For VESDE, the mean multiplier is 1 (forward process is x_t = x0 + sigma_t * eps).
    """
    return torch.ones_like(t)

def target_score(x_t: torch.Tensor, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Target score for VESDE.
    x_t = x0 + sigma(t) * eps
    score = (x_t - x0) / sigma(t)^2
    """
    sigma = sigma_t(t)
    return (x_t - x0) / sigma.pow(2)