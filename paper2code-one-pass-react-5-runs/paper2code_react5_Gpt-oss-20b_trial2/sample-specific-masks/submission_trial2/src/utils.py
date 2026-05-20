#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions used by the SMM training script.
"""

import numpy as np
import torch
import torch.nn.functional as F


def create_pad_mask(img_size: int, pad: int = 28) -> torch.Tensor:
    """
    Create a padding mask: 1 in the central region, 0 in the outer frame.
    The mask is broadcasted to 3 channels.
    """
    mask = torch.zeros(3, img_size, img_size, dtype=torch.float32)
    h_start = pad
    h_end = img_size - pad
    w_start = pad
    w_end = img_size - pad
    mask[:, h_start:h_end, w_start:w_end] = 1.0
    return mask


def create_narrow_mask(img_size: int, width: int = 28) -> torch.Tensor:
    """Mask covering the central region with a border width of `width`."""
    return create_pad_mask(img_size, pad=width)


def create_medium_mask(img_size: int, width: int = 56) -> torch.Tensor:
    """Mask covering the central region with a border width of `width`."""
    return create_pad_mask(img_size, pad=width)


def create_full_mask(img_size: int) -> torch.Tensor:
    """Full watermark: mask is all ones."""
    return torch.ones(3, img_size, img_size, dtype=torch.float32)