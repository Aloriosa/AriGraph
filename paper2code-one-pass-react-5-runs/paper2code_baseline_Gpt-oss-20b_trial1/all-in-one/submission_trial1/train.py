"""
Train a transformer‑based posterior estimator on the Two Moons simulator.
The trained model is saved to 'posterior_model.pt'.
"""

import os
import numpy as np
import torch
import sbi
from sbi import utils as sbiutils
from sbi import inference as sbii
from utils import generate_simulation_batch

def main():
    # Set random seeds for reproducibility
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Hyper‑parameters
    n_simulations = 20000   # number of training simulations
    batch_size = 1000
    n_epochs = 10

    # Generate simulation data
    print("Generating training data...")
    thetas, xs = generate_simulation_batch(n_simulations)
    # sbi expects theta in shape (n_samples, dim) and x in shape (n_samples, dim_x)
    data = (thetas, xs)

    # Define the simulator for sbi
    def simulator(theta, n):
        # theta: (batch, dim)
        # generate n samples for each theta
        rng = np.random.default_rng()
        thetas = theta
        xs = np.array([two_moons_simulator(t, seed=rng.integers(0, 1e9)) for t in thetas])
        return xs

    # Wrap the simulator to accept torch tensors
    def torch_simulator(theta, n):
        theta_np = theta.detach().cpu().numpy()
        xs = np.array([two_moons_simulator(t, seed=np.random.randint(0, 1e9)) for t in theta_np])
        return torch.tensor(xs, dtype=torch.float32)

    # Define the inference method - transformer posterior
    print("Setting up inference method...")
    density_estimator = sbii.posterior_estimator(
        density_estimator="transformer",
        device="cpu",
        density_estimator_kwargs={
            "hidden_features": 200,
            "n_layers": 4,
            "n_heads": 4,
            "dropout": 0.1,
        },
    )

    # Create the inference object
    inf = sbii.SBI(
        simulator=torch_simulator,
        density_estimator=density_estimator,
        device="cpu",
    )

    # Train the posterior
    print("Training posterior...")
    posterior = inf.train(
        data=data,
        training_batch_size=batch_size,
        epochs=n_epochs,
        verbose=True,
    )

    # Save the model
    out_dir = "output"
    os.makedirs(out_dir, exist_ok=True)
    torch.save(posterior, os.path.join(out_dir, "posterior_model.pt"))
    print(f"Posterior model saved to {os.path.join(out_dir, 'posterior_model.pt')}")

if __name__ == "__main__":
    main()