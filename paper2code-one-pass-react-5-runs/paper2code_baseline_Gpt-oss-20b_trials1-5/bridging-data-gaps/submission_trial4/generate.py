#!/usr/bin/env python
"""
Generate images using the fine‑tuned diffusion model.
"""

import os
import torch
from diffusers import UNet2DModel, DDPMScheduler
import torchvision.transforms.functional as TF
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
print("Loading fine‑tuned UNet...")
unet = UNet2DModel.from_pretrained("google/ddpm-cifar10-32").to(device)
unet.load_state_dict(torch.load('output/ckpt.pth', map_location=device))
unet.eval()

scheduler = DDPMScheduler.from_pretrained("google/ddpm-cifar10-32")

# Generate 100 images
num_samples = 100
samples = []

print("Generating samples...")
for i in tqdm(range(num_samples)):
    # start from random noise
    latents = torch.randn((1, 3, 32, 32), device=device)

    for t in reversed(range(scheduler.num_train_timesteps)):
        t_batch = torch.full((1,), t, device=device, dtype=torch.long)
        noise_pred = unet(latents, t_batch).sample
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    img = latents.squeeze().cpu()
    # Denormalize from (-1, 1) to (0, 1)
    img = (img + 1) / 2
    img = torch.clamp(img, 0, 1)
    samples.append(img)

# Save images
os.makedirs('output', exist_ok=True)
for idx, img in enumerate(samples):
    TF.to_pil_image(img).save(f'output/generated_{idx:03d}.png')

print(f"Saved {num_samples} images to ./output/")