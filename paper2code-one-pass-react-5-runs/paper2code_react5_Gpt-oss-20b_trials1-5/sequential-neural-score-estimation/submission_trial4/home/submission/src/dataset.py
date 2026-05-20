import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
from .utils import sigma_t, target_score_with_likelihood

class GaussianLinearDataset(Dataset):
    """
    Dataset for the Gaussian‑linear benchmark.
    Each sample: (theta_t, x_obs, t, target_score)
    """
    def __init__(self,
                 theta_0: torch.Tensor,
                 x_obs: torch.Tensor,
                 t: torch.Tensor,
                 cfg):
        """
        theta_0: (N, d) initial parameters
        x_obs: (d,) observation
        t: (N,) diffusion times
        """
        self.theta_0 = theta_0
        self.x_obs = x_obs
        self.t = t
        self.sigma_min = cfg["sigma_min"]
        self.sigma_max = cfg["sigma_max"]
        self.d = theta_0.shape[1]
        self.cfg = cfg

        # Pre‑compute theta_t and target scores
        sigma = sigma_t(t, self.sigma_min, self.sigma_max).unsqueeze(1)  # (N,1)
        noise = torch.randn_like(theta_0) * sigma
        self.theta_t = theta_0 + noise  # (N,d)

        self.target = target_score_with_likelihood(self.theta_t,
                                                   self.theta_0,
                                                   self.t,
                                                   self.x_obs,
                                                   cfg)  # (N,d)

    def __len__(self):
        return self.theta_t.shape[0]

    def __getitem__(self, idx):
        return self.theta_t[idx], self.x_obs, self.t[idx], self.target[idx]