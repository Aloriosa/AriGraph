"""
Sample from the trained posterior for a given observation.
The samples are saved to 'posterior_samples.csv'.
"""

import os
import numpy as np
import torch
import pandas as pd
from utils import two_moons_simulator

def main():
    # Load the trained posterior
    model_path = os.path.join("output", "posterior_model.pt")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Posterior model not found at {model_path}")

    posterior = torch.load(model_path)

    # Generate a test observation by sampling a true theta
    rng = np.random.default_rng(seed=123)
    true_theta = rng.uniform(-1.0, 1.0, size=(2,))
    true_x = two_moons_simulator(true_theta, seed=rng.integers(0, 1e9))
    print(f"True theta: {true_theta}")
    print(f"Observed x: {true_x}")

    # Sample posterior given the observation
    n_samples = 5000
    samples = posterior.sample(
        parameters=2,
        x=torch.tensor(true_x, dtype=torch.float32).unsqueeze(0),
        sample_shape=torch.Size([n_samples]),
    )
    samples = samples.detach().cpu().numpy()

    # Save samples to CSV
    out_dir = "output"
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(samples, columns=["theta1", "theta2"])
    csv_path = os.path.join(out_dir, "posterior_samples.csv")
    df.to_csv(csv_path, index=False)
    print(f"Posterior samples written to {csv_path}")

if __name__ == "__main__":
    main()