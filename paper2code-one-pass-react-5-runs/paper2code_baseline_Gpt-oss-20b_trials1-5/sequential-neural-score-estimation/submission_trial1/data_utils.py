import torch
from sklearn.datasets import make_moons


def simulator(theta: torch.Tensor, n_samples: int = 1, noise: float = 0.1) -> torch.Tensor:
    """
    Simple simulator for the Two‑Moons benchmark.
    Draws n_samples points from the two‑moons distribution and shifts
    them by the parameter vector θ (a 2‑dimensional shift).
    """
    X, _ = make_moons(n_samples=n_samples, noise=noise, random_state=None)
    X = torch.tensor(X, dtype=torch.float32)
    # Shift by θ
    X = X + theta
    return X