import math
import torch


def sigma_t(t: float, sigma_min: float, sigma_max: float) -> float:
    """
    Standard deviation σ_t of the variance‑exploding SDE:
        σ_t = σ_min * (σ_max / σ_min)^t
    """
    return sigma_min * (sigma_max / sigma_min) ** t