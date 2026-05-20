import torch
import torch.nn as nn

class AutoEncoderDescriptor(nn.Module):
    """
    Simple auto‑encoder used as a representation descriptor.
    It is trained to reconstruct the CLS token representation
    of a particular adapter. The reconstruction error is used
    to detect distribution shift.
    """
    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.encoder = nn.Linear(dim, hidden, bias=False)
        self.decoder = nn.Linear(hidden, dim, bias=False)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.encoder(x))
        return self.decoder(h)

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Mean‑squared reconstruction error for a batch of samples.
        """
        recon = self.forward(x)
        return torch.mean((x - recon) ** 2, dim=-1)  # shape [batch]