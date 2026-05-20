import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def mask_to_indices(mask: np.ndarray) -> np.ndarray:
    """Return indices where mask==1."""
    return np.where(mask == 1)[0]

def indices_to_mask(indices: np.ndarray, size: int) -> np.ndarray:
    """Create binary mask from indices."""
    mask = np.zeros(size, dtype=np.int32)
    mask[indices] = 1
    return mask