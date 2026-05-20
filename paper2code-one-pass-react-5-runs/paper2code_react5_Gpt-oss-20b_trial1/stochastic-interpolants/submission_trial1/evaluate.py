#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms

from utils import InpaintDataset, SuperResDataset, psnr
from model import SimpleUNet

def to_device(t, device):
    if isinstance(t, (list, tuple)):
        return [to_device(e, device) for e in t]
    return t.to(device)

def main():
    parser = argparse.ArgumentParser(description="Evaluate PSNR on saved samples")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model_path = Path(args.input_dir) / "best_model.pt"
    cond_ch = 1 if "inpaint" in args.input_dir else 3
    model = SimpleUNet(in_channels=3, cond_channels=cond_ch,
                       time_dim=64, base_channels=64).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load dataset
    if "inpaint" in args.input_dir:
        val_ds = InpaintDataset(split="test", transform=None)
    else:
        val_ds = SuperResDataset(split="test", transform=None)

    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4)

    psnr_vals = []
    with torch.no_grad():
        for x0, mask, target in val_loader:
            x0, mask, target = to_device([x0, mask, target], device)
            batch_size = x0.shape[0]

            # Continuous‑time integration with torchdiffeq
            from torchdiffeq import odeint

            def ode_func(t, y):
                cond = mask if isinstance(val_ds, InpaintDataset) else x0[:, :3, :, :]
                return model(y, t, cond)

            t_span = torch.linspace(0.0, 1.0, steps=100, device=device)
            y = odeint(ode_func, x0, t_span, method="dopri5")
            y_T = y[-1]
            y_T = torch.clamp(y_T, 0.0, 1.0)
            psnr_vals.append(psnr(y_T, target))

    mean_psnr = np.mean(psnr_vals)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.output_dir) / "psnr.txt", "w") as f:
        f.write(f"{mean_psnr:.2f}\n")
    print(f"Mean PSNR: {mean_psnr:.2f} dB")

if __name__ == "__main__":
    main()