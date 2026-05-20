import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class SyntheticStateDataset(Dataset):
    """
    A simple dataset of 2‑D states uniformly sampled from a unit square.
    """
    def __init__(self, num_states=1000, state_dim=2, seed=42):
        rng = np.random.default_rng(seed)
        self.states = rng.uniform(0.0, 1.0, size=(num_states, state_dim)).astype(np.float32)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx]