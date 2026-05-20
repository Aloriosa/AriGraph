"""
Generate images using the fine‑tuned adapter.
"""

import argparse
import os
import yaml
from pathlib import Path
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from diffusers import UNet2DModel, DDPMScheduler
from diffusers.utils import logging
from PIL import Image

from .adapter import TimeAdapter

logging.set_verbosity_error()

class EmptyDataset(Dataset):
    """A dummy dataset that yields a single random noise tensor."""
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return torch.randn(3, 256, 256)

def main(cfg_path: str):
    cfg = yaml.safe_load(open(cfg_path))
    gen_cfg = cfg["generate"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pre‑trained DDPM
    unet = UNet2DModel.from_pretrained("google/ddpm-celebahq-256").to(device)
    unet.eval()
    scheduler = DDPMScheduler.from_pretrained("google/ddpm-celebahq-256")
    num_timesteps = scheduler.config.num_train_timesteps

    # Load adapter
    adapter = TimeAdapter(num_timesteps).to(device)
    adapter.load_state_dict(torch.load("output/adapter.pt", map_location=device))
    adapter.eval()

    # Generation loop
    os.makedirs("output/generated_images", exist_ok=True)
    torch.manual_seed(gen_cfg["seed"])
    for i in tqdm(range(gen_cfg["num_samples"]), desc="Generating"):
        # Start from random noise
        img = torch.randn(1, 3, 256, 256, device=device)
        for t in reversed(range(num_timesteps)):
            t_batch = torch.full((1,), t, device=device, dtype=torch.long)

            # Predict noise
            eps_pred = unet(img, t_batch).sample

            # Apply adapter
            eps_pred = adapter(eps_pred, t_batch)

            # Update image
            img = scheduler.step(eps_pred, t, img).prev_sample
            if t == 0:
                break  # we only need a few steps for speed
        # Convert to PIL
        img = ((img.squeeze().cpu() + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)
        img = transforms.ToPILImage()(img)
        img.save(f"output/generated_images/gen_{i:02d}.png")

    print("Generation finished. Images saved to output/generated_images/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)