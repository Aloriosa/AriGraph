"""
Utility functions used across the project.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def ensure_dir(path: str):
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def save_numpy(filename: str, array: np.ndarray):
    """Save a NumPy array to disk."""
    np.save(filename, array)


def load_numpy(filename: str) -> np.ndarray:
    """Load a NumPy array from disk."""
    return np.load(filename)


class SimulatorDataset(Dataset):
    """
    PyTorch Dataset that contains simulated pairs (theta, x).
    """
    def __init__(self, thetas: np.ndarray, xs: np.ndarray):
        assert thetas.shape[0] == xs.shape[0]
        self.thetas = torch.from_numpy(thetas).float()
        self.xs = torch.from_numpy(xs).float()

    def __len__(self):
        return self.thetas.shape[0]

    def __getitem__(self, idx):
        return self.thetas[idx], self.xs[idx]