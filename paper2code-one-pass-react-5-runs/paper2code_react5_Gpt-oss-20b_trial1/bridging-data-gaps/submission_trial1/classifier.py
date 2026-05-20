import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding (as in Transformer)."""
    def __init__(self, dim: int, max_time: int = 1000):
        super().__init__()
        self.dim = dim
        self.max_time = max_time
        self.emb = nn.Embedding(max_time, dim)
        self.register_buffer('freqs', torch.arange(dim // 2, dtype=torch.float32))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,)
        t = t.unsqueeze(-1).float()  # (B,1)
        freqs = self.freqs / self.max_time  # (dim/2,)
        args = t * 2 * torch.pi * freqs  # (B, dim/2)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)
        return emb

class SourceTargetClassifier(nn.Module):
    """
    Simple CNN classifier that predicts whether a noised image
    comes from the source (CIFAR‑10) or the target (SVHN).
    """
    def __init__(self, in_channels: int = 3, time_emb_dim: int = 128, num_classes: int = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.t_emb = TimeEmbedding(time_emb_dim)
        self.fc = nn.Sequential(
            nn.Linear(128 * 32 * 32 + time_emb_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)  – noised image
            t: (B,)          – timestep (int)
        Returns:
            logits: (B, num_classes)
        """
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = h.view(h.size(0), -1)  # flatten
        t_emb = self.t_emb(t)      # (B, time_emb_dim)
        h = torch.cat([h, t_emb], dim=1)
        logits = self.fc(h)
        return logits