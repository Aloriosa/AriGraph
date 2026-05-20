#!/usr/bin/env python3
"""
Generate samples by integrating the learned velocity field
using a simple Euler solver.
"""

import argparse
import os
import torch
import numpy as np
from PIL import Image
from model import VelocityNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def euler_integration(net, x0, steps=100):
    """
    Integrate the probability flow ODE:
        dX_t = b_t(X_t) dt
    using Euler with fixed step size.
    """
    dt = 1.0 / steps
    x = x0.clone()
    for step in range(steps):
        t = torch.tensor(step / steps, dtype=x.dtype, device=device)
        b = net(x, t.reshape(1))
        x = x + dt * b
    return x


def main(args):
    # Load network
    net = VelocityNet().to(device)
    net.load_state_dict(torch.load("checkpoints/velocity.pth", map_location=device))
    net.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    for i in range(args.num_samples):
        # Start from a random Gaussian vector
        x0 = torch.randn(1, 784, device=device) * 1.0  # std=1

        # Integrate
        x1 = euler_integration(net, x0, steps=args.steps)

        # Convert to image
        img = x1.squeeze().cpu().numpy()
        img = np.clip(img, 0.0, 1.0)
        img = (img * 255).astype(np.uint8).reshape(28, 28)
        im = Image.fromarray(img, mode="L")
        im.save(os.path.join(args.output_dir, f"sample_{i:02d}.png"))
        print(f"Saved sample {i:02d} to {args.output_dir}")

    print("Sampling finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate samples")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="samples")
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    main(args)