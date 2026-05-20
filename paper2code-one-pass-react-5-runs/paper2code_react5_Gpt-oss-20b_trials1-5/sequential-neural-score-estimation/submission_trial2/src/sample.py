import argparse
import os
import numpy as np
import torch
from src.simulator import sample_reference
from src.diffusion import VESDE
from src.models import ScoreNetwork
from src.train_npse import sample_posterior


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load model
    score_net = ScoreNetwork(theta_dim=1, x_dim=1).to(device)
    score_net.load_state_dict(torch.load(args.model, map_location=device))
    score_net.eval()

    # Diffusion
    diffusion = VESDE(sigma_min=0.01, sigma_max=10.0)

    # Observation (toy example)
    x_obs = torch.tensor([0.5], dtype=torch.float32, device=device).unsqueeze(0)

    # Generate posterior samples
    samples = sample_posterior(
        x_obs, args.num_samples, score_net, diffusion, device
    )
    samples = samples.cpu().numpy()

    # Save
    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, "posterior_samples.npy"), samples)
    print(f"Saved {args.num_samples} posterior samples to {args.out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample from TSNPSE posterior")
    parser.add_argument("--num-samples", type=int, default=5000, help="Number of posterior samples")
    parser.add_argument("--model", type=str, required=True, help="Path to trained score network")
    parser.add_argument("--out-dir", type=str, default="output", help="Output directory")
    args = parser.parse_args()
    main(args)