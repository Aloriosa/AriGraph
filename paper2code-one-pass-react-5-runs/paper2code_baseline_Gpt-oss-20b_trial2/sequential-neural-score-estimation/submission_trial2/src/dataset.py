import numpy as np
import torch
from .utils import to_tensor

def gaussian_linear_prior(batch: int, dim: int = 10, scale: float = 0.1):
    """Prior p(θ) = N(0, scale² I)."""
    return np.random.normal(0, scale, size=(batch, dim))

def gaussian_linear_likelihood(theta: np.ndarray, scale: float = 0.1):
    """Likelihood p(x | θ) = N(x | θ, scale² I)."""
    return np.random.normal(theta, scale)

def toy_prior(batch: int, dim: int = 1, a: float = -1.0, b: float = 1.0):
    """Uniform prior on [-1,1] for toy example."""
    return np.random.uniform(a, b, size=(batch, dim))

def toy_likelihood(theta: np.ndarray):
    """Simulate x from the "two moons" toy model in the paper."""
    # For the toy, we only need a single observation: the word "strawberry".
    # Therefore the likelihood is deterministic: x = "strawberry".
    # We encode it as a fixed one‑hot vector of length 1 (value 1.0).
    return np.ones_like(theta)  # placeholder

class GaussianLinearDataset:
    """
    Dataset for the Gaussian‑Linear benchmark.
    Generates (θ₀, x) pairs on the fly.
    """
    def __init__(self, n: int, dim: int =10, scale: float =0.1):
        self.n = n
        self.dim = dim
        self.scale = scale

    def sample(self, batch: int) -> Tuple[torch.Tensor, torch.Tensor]:
        theta0 = gaussian_linear_prior(batch, self.dim, self.scale)
        x = gaussian_linear_likelihood(theta0, self.scale)
        return to_tensor(theta0), to_tensor(x)

class ToyDataset:
    """
    Toy dataset for the toy counting example.
    Generates (θ₀, x) pairs with θ₀ ∈ [−1,1] and x fixed.
    """
    def __init__(self, n: int):
        self.n = n

    def sample(self, batch: int) -> Tuple[torch.Tensor, torch.Tensor]:
        theta0 = toy_prior(batch, dim=1)
        # x is constant: the word "strawberry" encoded as 1.0
        x = np.ones_like(theta0)
        return to_tensor(theta0), to_tensor(x)