#!/usr/bin/env python3
"""
Train a simple conditional diffusion model (probability‑flow ODE) on CIFAR‑10
using a data‑dependent coupling for in‑painting.

The loss is a simple L2 between the predicted velocity and the true
velocity, which is the minimizer of the quadratic objective in the paper.
"""

import argparse
import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from tqdm import tqdm

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_mask(batch_size, img_size=32, mask_size=8):
    """
    Return a mask of shape (B, 1, H, W) where a random square block
    of size (mask_size, mask_size) is set to 0 (masked) and the rest to 1.
    """
    masks = torch.ones(batch_size, 1, img_size, img_size)
    for i in range(batch_size):
        top = random.randint(0, img_size - mask_size)
        left = random.randint(0, img_size - mask_size)
        masks[i, 0, top:top+mask_size, left:left+mask_size] = 0.0
    return masks

# --------------------------------------------------------------------------- #
# Model definition (a tiny U‑Net‑like CNN)
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
        # Encoder
        e1 = self.act(self.enc1(x))
        e2 = self.act(self.enc2(e1))
        e3 = self.act(self.enc3(e2))
        # Decoder (simple skip connections)
        d1 = self.act(self.dec1(e3 + e2))
        d2 = self.act(self.dec2(d1 + e1))
        out = self.dec3(d2)
        return out

# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #
def train(
    epochs: int,
    batch_size: int,
    output_dir: str,
    lr: float = 2e-4,
    device: torch.device = None,
):
    set_seed(42)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data loader
    transform = transforms.Compose([
        transforms.ToTensor(),  # [0,1]
    ])
    dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    # Model
    model = SimpleUNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}")
        for imgs, _ in pbar:
            imgs = imgs.to(device)  # [B,3,H,W] in [0,1]

            # Create mask and base sample
            masks = get_mask(imgs.shape[0], img_size=32, mask_size=8).to(device)
            # Base sample: observed pixels keep original, masked pixels are random noise
            noise = torch.randn_like(imgs)
            x0 = masks * imgs + (1.0 - masks) * noise

            # Random time t ~ Uniform(0,1)
            t = torch.rand(imgs.shape[0], device=device)
            alpha = t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)            # shape (B,1,1,1)
            beta  = (1.0 - t).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

            # Interpolated image I_t
            It = alpha * x0 + beta * imgs

            # True velocity: dotI_t = d/dt(alpha)*x0 + d/dt(beta)*imgs
            # d(alpha)/dt = 1, d(beta)/dt = -1
            dot_it = x0 - imgs

            # Predict velocity
            pred = model(It)

            # Loss: mean squared error between predicted and true velocity
            loss = nn.functional.mse_loss(pred, dot_it)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=loss.item())

        # Save checkpoint
        torch.save(model.state_dict(), os.path.join(output_dir, f"model_epoch{epoch}.pth"))

    # Final model
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pth"))
    print(f"Training finished. Model saved to {os.path.join(output_dir, 'model.pth')}")

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train data‑dependent diffusion model.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch", type=int, default=128, help="Batch size.")
    parser.add_argument("--output_dir", type=str, default="./model", help="Folder to save checkpoints.")
    args = parser.parse_args()
    train(args.epochs, args.batch, args.output_dir)