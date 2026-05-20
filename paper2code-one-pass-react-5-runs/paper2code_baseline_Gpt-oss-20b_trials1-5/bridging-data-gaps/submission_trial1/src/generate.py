#!/usr/bin/env python
"""
Generate images from the fine‑tuned UNet (DDPM) using a simple
reverse diffusion loop.

The script loads the model checkpoint, performs 1000 denoising steps,
and saves the final images as PNG files.
"""

import argparse
import os
from pathlib import Path

import torch
import torchvision.transforms as T
from torchvision.utils import save_image
from diffusers import UNet2DModel, DDPMScheduler
from tqdm import tqdm

def load_model(model_path: str, device):
    unet = UNet2DModel.from_pretrained("google/ddpm-cifar10-32")
    unet.load_state_dict(torch.load(model_path, map_location=device))
    unet.to(device)
    unet.eval()
    return unet

def sample(unet, scheduler, num_samples, device):
    """
    Sample images from the DDPM model.
    """
    shape = (num_samples, 3, 32, 32)

    # Start from random noise
    latents = torch.randn(shape, device=device)

    for i in tqdm(reversed(range(scheduler.num_inference_steps)), desc="Sampling"):
        t = torch.full((num_samples,), i, device=device, dtype=torch.long)
        # Predict noise
        noise_pred = unet(latents, t).sample
        # Update latents
        latents = scheduler.step(noise_pred, i, latents).prev_sample

    # Denormalize from [-1,1] to [0,1]
    latents = (latents + 1) / 2
    latents = torch.clamp(latents, 0, 1)
    return latents

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    unet = load_model(args.model_path, device)
    scheduler = DDPMScheduler(num_train_timesteps=1000)

    samples = sample(unet, scheduler, args.num_images, device)

    os.makedirs(args.output_dir, exist_ok=True)
    for i in range(samples.shape[0]):
        img = samples[i]
        save_path = os.path.join(args.output_dir, f"sample_{i+1}.png")
        save_image(img, save_path)
        print(f"Saved {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the fine‑tuned model checkpoint")
    parser.add_argument("--num_images", type=int, default=5,
                        help="Number of images to generate")
    parser.add_argument("--output_dir", type=str, default="output/generated",
                        help="Directory to save generated images")
    args = parser.parse_args()
    main(args)