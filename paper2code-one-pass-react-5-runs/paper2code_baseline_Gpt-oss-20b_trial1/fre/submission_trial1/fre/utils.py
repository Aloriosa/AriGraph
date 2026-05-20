import torch
import math
import numpy as np

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)

def gaussian_noise(tensor, std=0.0):
    if std == 0.0:
        return tensor
    noise = torch.randn_like(tensor) * std
    return tensor + noise