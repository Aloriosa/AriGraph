import torch
import torch.nn.functional as F

def patchwise_interpolate(mask, scale_factor):
    """
    Upscales a mask by copying each pixel into a scale_factor × scale_factor block.
    The operation is non‑differentiable (no gradient).
    mask: Tensor of shape (B, C, h, w)
    scale_factor: int
    Returns Tensor of shape (B, C, h*scale_factor, w*scale_factor)
    """
    B, C, h, w = mask.shape
    # Expand to (B, C, h, 1, w, 1)
    mask = mask.unsqueeze(3).unsqueeze(5)
    # Repeat along new dims
    mask = mask.repeat(1, 1, 1, scale_factor, 1, scale_factor)
    # Reshape to 4‑D
    mask = mask.view(B, C, h*scale_factor, w*scale_factor)
    return mask