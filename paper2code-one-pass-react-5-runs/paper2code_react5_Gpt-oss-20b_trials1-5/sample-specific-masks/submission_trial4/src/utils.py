import torch
import random
import numpy as np

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_mask(mask_size: str, device: torch.device, shape: torch.Size):
    """
    Create a binary mask for the input batch.
    :param mask_size: 'full', 'medium', or 'narrow'
    :param device: torch device
    :param shape: (B, C, H, W)
    :return: mask tensor of shape (B, C, H, W) on the given device
    """
    B, C, H, W = shape
    if mask_size == "full":
        return torch.ones((B, C, H, W), device=device)
    elif mask_size == "medium":
        size = H // 4  # 56 for 224
        top = (H - size) // 2
        left = (W - size) // 2
        mask = torch.zeros((B, C, H, W), device=device)
        mask[:, :, top:top+size, left:left+size] = 1.0
        return mask
    elif mask_size == "narrow":
        size = H // 8  # 28 for 224
        top = (H - size) // 2
        left = (W - size) // 2
        mask = torch.zeros((B, C, H, W), device=device)
        mask[:, :, top:top+size, left:left+size] = 1.0
        return mask
    else:
        raise ValueError(f"Unsupported mask_size: {mask_size}")