#!/usr/bin/env python
import argparse
import numpy as np
import torch
from src.npse import NPSE

def main():
    parser = argparse.ArgumentParser(description="Sample from a trained NPSE model")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to the PyTorch state_dict file")
    parser.add_argument("--n_samples", type=int, default=2000,
                        help="Number of posterior samples to draw")
    parser.add_argument("--n_steps", type=int, default=1000,
                        help="Number of ODE integration steps")
    args = parser.parse_args()

    # Load model
    # we need to infer dim_theta and dim_x – here we read them from the file name
    # (this is a simple convention used in the demo)
    if "gaussian_linear" in args.model:
        dim_theta = dim_x = 10
    elif "toy" in args.model:
        dim_theta = dim_x = 1
    else:
        raise ValueError("Unknown model name – cannot infer dimensionality")

    device = "cpu"
    model = NPSE(dim_theta, dim_x, device=device)
    model.load(args.model)

    samples = model.sample(args.n_samples, n_steps=args.n_steps)
    np.save("posterior_samples.npy", samples)
    print(f"Saved {args.n_samples} samples to posterior_samples.npy")

if __name__ == "__main__":
    main()