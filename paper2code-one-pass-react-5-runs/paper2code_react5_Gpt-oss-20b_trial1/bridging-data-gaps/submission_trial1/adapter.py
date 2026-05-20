import torch
import torch.nn as nn

class Adapter(nn.Module):
    """
    Lightweight 1×1‑conv residual adapter added to the diffusion model.
    The adapter takes the noised image x_t as input and outputs a residual
    that is added to the predicted noise ε̂ from the frozen UNet.
    """
    def __init__(self, channels: int = 3, bottleneck_ratio: int = 4):
        super().__init__()
        bottleneck = channels // bottleneck_ratio
        self.net = nn.Sequential(
            nn.Conv2d(channels, bottleneck, kernel_size=1, bias=False),
            nn.BatchNorm2d(bottleneck),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck, channels, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)