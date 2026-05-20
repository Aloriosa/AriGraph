import torch
import torch.nn.functional as F


def patch_interpolate(mask: torch.Tensor,
                      target_size: int,
                      patch_size: int) -> torch.Tensor:
    """
    Patch‑wise interpolation: each pixel of `mask` is repeated into a
    square patch of size `patch_size × patch_size` to reach the
    `target_size`.

    Parameters
    ----------
    mask : torch.Tensor
        Shape (B, C, h', w'), values in [0,1].
    target_size : int
        Desired spatial dimension (H = W = target_size).
    patch_size : int
        Size of the patch (must be a power of two).

    Returns
    -------
    torch.Tensor
        Shape (B, C, target_size, target_size).
    """
    _, _, h, w = mask.shape
    if h == target_size and w == target_size:
        return mask

    # Use nearest‑neighbor upsampling by the required scale factor
    scale = target_size // h
    if scale * h != target_size or scale * w != target_size:
        raise ValueError(
            f"Target size {target_size} is not a multiple of mask size "
            f"{h}x{w}.")
    return F.interpolate(mask, scale_factor=scale, mode="nearest")