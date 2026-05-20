import torchvision.transforms.functional as TF
import random
import torch

def gaussian_noise(img, mean=0.0, std=0.1):
    noise = torch.randn_like(img) * std + mean
    return torch.clamp(img + noise, 0.0, 1.0)

def blur(img, radius=2):
    return TF.gaussian_blur(img, kernel_size=(radius, radius), sigma=(0.1, 2.0))

def contrast(img, factor=0.5):
    return TF.adjust_contrast(img, factor)

def random_corruption(img, level=1):
    """Apply a random corruption from the set used in ImageNet‑C."""
    funcs = [gaussian_noise, blur, contrast]
    func = random.choice(funcs)
    return func(img, std=0.1*level, radius=2*level, factor=0.5+0.5*level)