import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class GaussianLinearSimulator:
    """
    Simple Gaussian‑linear simulator.
    Parameters:
        dim  : dimensionality of theta (and consequently of x)
        noise_std : observation noise std (default 0.1)
    """
    def __init__(self, dim: int = 5, noise_std: float = 0.1):
        self.dim = dim
        self.noise_std = noise_std

    def __call__(self, theta: np.ndarray) -> np.ndarray:
        """
        Generate observation x given theta.
        """
        return theta + self.noise_std * np.random.randn(self.dim)

class GaussianLinearDataset(Dataset):
    """
    Torch dataset that returns (theta, x) pairs.
    """
    def __init__(self, simulator: GaussianLinearSimulator, n_samples: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.theta = rng.normal(0, np.sqrt(0.1), size=(n_samples, simulator.dim))
        self.x = np.array([simulator(t) for t in self.theta])

    def __len__(self):
        return len(self.theta)

    def __getitem__(self, idx):
        return torch.tensor(self.theta[idx], dtype=torch.float32), \
               torch.tensor(self.x[idx], dtype=torch.float32)