#!/usr/bin/env python3
"""
Generate samples from the trained model by integrating the learned
probability‑flow ODE from t=0 to t=1.

The base distribution is defined by a masked image + random noise in
the masked region, exactly as in training.
"""

import argparse
import os
import random

import torch
import torch.nn as nn
from torchvision import datasets, transforms, utils
import numpy as np
from tqdm import tqdm

# --------------------------------------------------------------------------- #
# Reuse the same model definition
# --------------------------------------------------------------------------- #
class SimpleUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, hidden=64):
        super().__init__()
        self.enc1 = nn.Conv2d(in_channels, hidden, 3, padding=1)
        self.enc2 = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.enc3 = nn.Conv2d(hidden, hidden, 3, padding=1)

        self.dec1 = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.dec2 = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.dec3 = nn.Conv2d(hidden, out_channels, 3, padding=1)

        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        e1 = self.act(self.enc1(x))
        e2 = self.act(self.enc2(e1))
        e3 = self.act(self.enc3(e2))
        d1 = self.act(self.dec1(e3 + e2))
        d2 = self.act(self.dec2(d1 + e1))
        out = self.dec3(d2)
        return out

# --------------------------------------------------------------------------- #
# Sampling utilities
# --------------------------------------------------------------------------- #
def get_mask(batch_size, img_size=32, mask_size=8):
    masks = torch.ones(batch_size, 1, img_size, img_size)
    for i in range(batch_size):
        top = random.randint(0, img_size - mask_size)
        left = random.randint(0, img_size - mask_size)
        masks[i, 0, top:top+mask_size, left:left+mask_size] = 0.0
    return masks

# --------------------------------------------------------------------------- #
# Main sampling routine
# --------------------------------------------------------------------------- #
def sample(
    model_path: str,
    num_samples: int,
    output_dir: str,
    device: torch.device = None,
    steps: int = 50,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = SimpleUNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    os.makedirs(output_dir, exist_ok=True)

    # For reproducibility
    torch.manual_seed(42)

    # Create a dummy image (all zeros) – we only need the mask
    dummy = torch.zeros((num_samples, 3, 32, 32), device=device)
    masks = get_mask(num_samples, img_size=32, mask_size=8).to(device)

    # Base sample: observed pixels are zero (unknown), masked pixels are random noise
    noise = torch.randn_like(dummy)
    x0 = masks * dummy + (1.0 - masks) * noise

    # Euler integration
    dt = 1.0 / steps
    x = x0.clone()
    with torch.no_grad():
        for _ in tqdm(range(steps), desc="Sampling ODE"):
            vel = model(x)
            x = x + dt * vel

    # Clip to [0,1] and convert to uint8
    x = torch.clamp(x, 0.0, 1.0)
    grid = utils.make_grid(x, nrow=4, normalize=True)
    utils.save_image(grid, os.path.join(output_dir, "samples.png"))
    print(f"Samples saved to {os.path.join(output_dir, 'samples.png')}")

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate samples from the trained model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model .pth file.")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to generate.")
    parser.add_argument("--output_dir", type=str, default="./samples", help="Folder to save images.")
    args = parser.parse_args()
    sample(args.model_path, args.num_samples, args.output_dir)