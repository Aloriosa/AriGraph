"""
Training script for the simplified DPMs-ANT procedure.
It loads a pre‑trained DDPM (trained on FFHQ),
adds a TimeAdapter, and fine‑tunes only the adapter on a small
target dataset (10 images).
"""

import argparse
import os
import random
import yaml
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm

from diffusers import UNet2DModel, DDPMScheduler
from diffusers.utils import logging

from .adapter import TimeAdapter

logging.set_verbosity_error()

def load_config(cfg_path: str):
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)

def get_dataloader(data_dir: str, batch_size: int, seed: int = 42):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    # Ensure we only keep 10 images if more are present
    dataset = torch.utils.data.Subset(dataset, list(range(min(10, len(dataset)))))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    return loader

def main(cfg_path: str):
    cfg = load_config(cfg_path)
    train_cfg = cfg["train"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pre‑trained DDPM (FFHQ 256)
    print("Loading pre‑trained DDPM model...")
    unet = UNet2DModel.from_pretrained("google/ddpm-celebahq-256").to(device)
    unet.eval()  # freeze parameters
    scheduler = DDPMScheduler.from_pretrained("google/ddpm-celebahq-256")

    num_timesteps = scheduler.config.num_train_timesteps
    adapter = TimeAdapter(num_timesteps).to(device)

    optimizer = torch.optim.Adam(adapter.parameters(), lr=train_cfg["learning_rate"])

    # Prepare dataset
    data_dir = Path("data/target")
    loader = get_dataloader(data_dir, train_cfg["batch_size"])

    # Training loop
    total_steps = train_cfg["num_steps"]
    step = 0
    print("Starting training...")
    for epoch in range(train_cfg["epochs"]):
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}"):
            images, _ = batch  # images: (B, C, H, W)
            images = images.to(device)

            # Sample random timesteps
            t = torch.randint(0, num_timesteps, (images.shape[0],), device=device)

            # Sample noise
            noise = torch.randn_like(images, device=device)

            # Compute x_t
            sqrt_alpha_bar = torch.sqrt(scheduler.alphas_cumprod[t]).view(-1, 1, 1, 1)
            sqrt_one_minus_alpha_bar = torch.sqrt(1 - scheduler.alphas_cumprod[t]).view(-1, 1, 1, 1)
            x_t = sqrt_alpha_bar * images + sqrt_one_minus_alpha_bar * noise

            # Predict noise via UNet
            eps_pred = unet(x_t, t).sample

            # Adversarial noise selection (inner maximization)
            eps_adv = noise.clone().detach().requires_grad_(True)
            for _ in range(train_cfg["adversarial_steps"]):
                # Compute loss w.r.t noise
                loss_adv = F.mse_loss(eps_adv, eps_pred)
                loss_adv.backward()
                # Gradient ascent
                eps_adv = eps_adv + train_cfg["adversarial_lr"] * eps_adv.grad
                # Normalize to keep it ~N(0,1)
                eps_adv = (eps_adv - eps_adv.mean()) / (eps_adv.std() + 1e-6)
                eps_adv = eps_adv.detach().requires_grad_(True)

            # Use the worst‑case noise for the actual loss
            loss = F.mse_loss(eps_adv, eps_pred)

            # Backpropagate through adapter
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            if step >= total_steps:
                break
        if step >= total_steps:
            break

    # Save adapter
    Path("output").mkdir(parents=True, exist_ok=True)
    torch.save(adapter.state_dict(), "output/adapter.pt")
    print(f"Training finished. Adapter saved to output/adapter.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)