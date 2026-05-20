#!/usr/bin/env python
"""
Fine‑tune a pre‑trained DDPM (google/ddpm-cifar10-32) on 10 images.

The script:
  - Loads the first `num_samples` images from the specified dataset.
  - Adds a small adapter to every UNet block.
  - Uses a simple MSE loss between predicted noise and true noise,
    augmented with a similarity‑guided term (the negative of the
    gradient of a binary classifier that distinguishes source vs.
    target).  For simplicity, we use a dummy classifier that
    outputs zeros; this keeps the code short and deterministic.
"""

import argparse
import os
import random
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from diffusers import UNet2DModel, DDPMScheduler
from diffusers.optimization import get_scheduler
from datasets import load_dataset
from tqdm import tqdm
import torchvision.transforms as T

from src.adapter import Adapter

# ------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------------
# Dummy similarity classifier
# ------------------------------------------------------------------
class DummyClassifier(nn.Module):
    """
    Returns a constant gradient of zeros.  In the original paper,
    a real binary classifier distinguishing source vs. target
    would be trained.  Here we keep it simple.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x shape: (batch, channels, height, width)
        return torch.zeros_like(x, device=x.device)

# ------------------------------------------------------------------
# Main training routine
# ------------------------------------------------------------------
def main(args):
    set_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    # Load pre‑trained DDPM on CIFAR‑10
    print("Loading pre‑trained DDPM model...")
    unet = UNet2DModel.from_pretrained("google/ddpm-cifar10-32")
    unet.to(device)

    # Add adapters to every block
    for name, module in unet.named_modules():
        if isinstance(module, nn.Conv2d):
            # Wrap the conv with an adapter
            adapter = Adapter(module.out_channels).to(device)
            # Replace the module with a sequential of conv + adapter
            new_module = nn.Sequential(module, adapter)
            parent = dict(unet.named_modules())[name.rpartition('.')[0]]
            setattr(parent, name.rpartition('.')[2], new_module)

    # Freeze original weights
    for param in unet.parameters():
        param.requires_grad = False
    # Only adapters are trainable
    trainable_params = list(unet.parameters())

    # Scheduler
    scheduler = DDPMScheduler(num_train_timesteps=1000)

    # Optimizer
    optimizer = optim.AdamW(trainable_params, lr=args.lr)

    # Data
    print(f"Downloading {args.dataset} dataset...")
    dataset = load_dataset(args.dataset, split="train")
    indices = list(range(args.num_samples))
    subset = Subset(dataset, indices)
    transform = T.Compose([
        T.ToTensor(),          # [0,1]
        T.Normalize((0.5, 0.5, 0.5),
                    (0.5, 0.5, 0.5))  # to [-1,1]
    ])

    def collate_fn(batch):
        imgs = [transform(item["image"]) for item in batch]
        return torch.stack(imgs)

    loader = DataLoader(subset, batch_size=args.batch_size,
                        shuffle=True, collate_fn=collate_fn)

    # Dummy classifier
    classifier = DummyClassifier().to(device)

    # Training loop
    metrics = []
    for epoch in range(args.epochs):
        loss_epoch = 0.0
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            batch = batch.to(device)

            # Sample random timestep
            t = torch.randint(0, scheduler.alphas_cumprod.shape[0],
                              (batch.size(0),), device=device).long()

            # Generate noise and noisy image
            noise = torch.randn_like(batch)
            alpha_cumprod = scheduler.alphas_cumprod[t].view(-1, 1, 1, 1)
            sqrt_alpha = torch.sqrt(alpha_cumprod)
            sqrt_one_minus_alpha = torch.sqrt(1 - alpha_cumprod)

            x_t = sqrt_alpha * batch + sqrt_one_minus_alpha * noise

            # Predict noise
            with torch.no_grad():
                pred_noise = unet(x_t, t).sample

            # Similarity‑guided term (dummy)
            sim_grad = classifier(x_t)  # shape matches noise

            # Loss: MSE + gamma * ||sim_grad||
            loss = nn.functional.mse_loss(pred_noise, noise)
            loss += args.gamma * torch.mean(sim_grad ** 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_epoch += loss.item() * batch.size(0)

        avg_loss = loss_epoch / len(loader.dataset)
        metrics.append(avg_loss)
        with open(os.path.join(args.output_dir, "metrics.txt"), "a") as f:
            f.write(f"epoch {epoch+1}, loss {avg_loss:.6f}\n")
        print(f"Epoch {epoch+1} finished. Avg loss: {avg_loss:.6f}")

    # Save the fine‑tuned UNet
    torch.save(unet.state_dict(), os.path.join(args.output_dir, "model.pt"))
    print(f"Model saved to {os.path.join(args.output_dir, 'model.pt')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar10",
                        help="Dataset name (used by HuggingFace datasets)")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of images to use for fine‑tuning")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="Weight for similarity‑guided term")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Output directory")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)