#!/usr/bin/env python3
"""
evaluate.py – generate samples from the fine‑tuned UNet and compute
FID against the 10‑shot target set.

The script:
  1. Loads the checkpointed UNet and classifier.
  2. Generates 500 samples using DDPM sampling with 1000 steps.
  3. Computes FID between generated samples and target images.
  4. Saves generated images to ./samples and prints the FID.
"""

import os
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Diffusers imports
from diffusers import UNet2DModel, DDPMScheduler

# Torchmetrics for FID
from torchmetrics.image.fid import FrechetInceptionDistance

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ------------------------------------------------------------- #
# Helper functions
# ------------------------------------------------------------- #
def load_target_images():
    """Return 10‑shot CIFAR10 images of target class (cat)."""
    from torchvision.datasets import CIFAR10
    transform = T.Compose([T.ToTensor()])
    train_ds = CIFAR10(root="data", train=True, download=True, transform=transform)
    target_class = 3  # cat
    indices = [i for i, (_, label) in enumerate(train_ds) if label == target_class][:10]
    imgs = [train_ds[i][0] for i in indices]
    return torch.stack(imgs)  # shape [10,3,32,32]

def generate_samples(unet, scheduler, device, n_samples=500, seed=SEED):
    """Generate samples using DDPM sampling."""
    torch.manual_seed(seed)
    unet.eval()
    samples = []
    batch_size = 64
    n_batches = (n_samples + batch_size - 1) // batch_size
    for i in tqdm(range(n_batches), desc="Sampling"):
        B = min(batch_size, n_samples - i * batch_size)
        # start from random noise in [-1,1]
        latents = torch.randn(B, 3, 32, 32, device=device)

        for t in reversed(range(scheduler.num_train_timesteps)):
            t_batch = torch.full((B,), t, dtype=torch.long, device=device)
            eps = unet(latents, t_batch).sample
            latents = scheduler.step(eps, t, latents).prev_sample

        samples.append(latents.cpu())

    samples = torch.cat(samples, dim=0)
    samples = torch.clamp(samples, -1.0, 1.0)  # keep in [-1,1]
    return samples

# ------------------------------------------------------------- #
# Main
# ------------------------------------------------------------- #
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpointed UNet
    unet = UNet2DModel.from_pretrained("google/ddpm-cifar10-32").to(device)
    scheduler = DDPMScheduler.from_pretrained("google/ddpm-cifar10-32")
    ckpt = torch.load("checkpoints/unet.pt", map_location=device)
    unet.load_state_dict(ckpt)

    # Generate samples
    samples = generate_samples(unet, scheduler, device, n_samples=500)

    # Load target images and normalise to [-1,1]
    target_imgs = load_target_images().to(device)
    target_imgs = target_imgs * 2 - 1  # To [-1,1]

    # Convert to [0,1] for FID
    fid = FrechetInceptionDistance().to(device)
    fid.update((samples + 1) / 2, real=False)
    fid.update((target_imgs + 1) / 2, real=True)
    print(f"FID: {fid.compute().item():.2f}")

    # Save generated images
    Path("samples").mkdir(parents=True, exist_ok=True)
    inv_norm = T.Normalize(mean=[-1, -1, -1], std=[2, 2, 2])  # [-1,1] -> [0,1]
    to_pil = T.ToPILImage()
    for i, img in enumerate(samples):
        img_denorm = inv_norm(img)
        pil = to_pil(img_denorm)
        pil.save(f"samples/sample_{i:04d}.png")