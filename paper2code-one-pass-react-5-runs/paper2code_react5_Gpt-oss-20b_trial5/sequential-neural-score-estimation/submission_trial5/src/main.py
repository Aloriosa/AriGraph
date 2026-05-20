import torch
from .tsnpse import TSNPSE
from .simulator import GaussianLinearSimulator
from .utils import set_seed

def main():
    set_seed(42)
    device = 'cpu'
    sim = GaussianLinearSimulator(dim=10, prior_std=0.1, likelihood_std=0.1, seed=0)

    tsnpse = TSNPSE(
        simulator=sim,
        rounds=5,
        total_simulations=5000,
        batch_size=256,
        lr=1e-4,
        device=device,
        eps_hpr=0.05,
        kde_bandwidth=0.1,
        seed=42
    )
    tsnpse.run()

if __name__ == "__main__":
    main()