import os
import math
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, models
from PIL import Image

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def to_tensor(img):
    return transforms.ToTensor()(img)

def to_pil(tensor):
    return transforms.ToPILImage()(tensor.clamp(0,1))

def save_image_grid(tensors, nrow, path):
    grid = torchvision.utils.make_grid(tensors, nrow=nrow)
    torchvision.utils.save_image(grid, path)

def time_embed(t, dim=64):
    """Sinusoidal time embedding as in DDPM."""
    half = dim // 2
    emb = math.log(10000) / (half - 1)
    emb = torch.exp(torch.arange(half, device=t.device) * -emb)
    emb = t[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    return emb