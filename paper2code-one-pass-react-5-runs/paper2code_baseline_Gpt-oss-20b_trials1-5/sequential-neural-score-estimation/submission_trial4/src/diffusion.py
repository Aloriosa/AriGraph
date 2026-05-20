import torch
import numpy as np

def sigma_t(t, sigma_min=0.01, sigma_max=1.0):
    """
    VE‑SDE noise scale as a function of time t in [0,1].
    """
    return sigma_min * (sigma_max / sigma_min) ** t

def noise_schedule(num_steps):
    """
    Return a list of time points from 1 to 0.
    """
    return np.linspace(1.0, 0.0, num_steps + 1)