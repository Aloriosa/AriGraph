"""
Utility functions used by the training script.
"""

import math
import torch
import torch.nn.functional as F

def normalize_noise(noise, eps=1e-12):
    """
    Normalize a noise tensor to have zero mean and unit std.
    """
    mean = noise.mean(dim=[1, 2, 3], keepdim=True)
    std = noise.std(dim=[1, 2, 3], keepdim=True) + eps
    return (noise - mean) / std

def get_alpha_cumprod(scheduler):
    """
    Return the alphas_cumprod tensor from a DDPMScheduler.
    """
    return scheduler.alphas_cumprod.to(scheduler.device)

def get_beta(scheduler, t):
    """
    Return beta_t for a given timestep t.
    """
    alphas_cumprod = get_alpha_cumprod(scheduler)
    alpha_t = alphas_cumprod[t]
    alpha_next = alphas_cumprod[t-1] if t > 0 else 1.0
    return (1 - alpha_t) / (1 - alpha_next)