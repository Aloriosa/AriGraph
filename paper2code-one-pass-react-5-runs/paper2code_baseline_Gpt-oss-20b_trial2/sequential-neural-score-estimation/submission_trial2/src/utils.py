import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from functools import partial
from typing import Tuple, List

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def to_tensor(x: np.ndarray, device: str = "cpu") -> torch.Tensor:
    return torch.from_numpy(x.astype(np.float32)).to(device)

def sinusoidal_embedding(t: torch.Tensor, dim: int = 64) -> torch.Tensor:
    """
    Sinusoidal positional encoding for time variable t in [0,1].
    """
    # t shape: (batch,)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim // 2, dtype=torch.float32) / dim))
    sinusoid_inp = t[:, None] * inv_freq[None, :]
    pos_enc = torch.cat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=-1)
    return pos_enc  # (batch, dim)