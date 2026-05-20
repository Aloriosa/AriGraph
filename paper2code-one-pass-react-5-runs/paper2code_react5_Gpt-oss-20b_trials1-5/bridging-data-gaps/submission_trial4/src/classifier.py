import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryClassifier(nn.Module):
    """
    Simple CNN that classifies a noised image as source (0) or target (1).
    It is trained on the same timesteps that the diffusion model uses.
    """
    def __init__(self, in_channels=3, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden, 2)
        )
        self.to('cuda')

    def forward(self, x, timesteps):
        # timesteps are not used directly; we could embed them if desired
        return self.net(x)

    def train_classifier(self, source_loader, target_loader, epochs=2, lr=1e-3):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        self.train()
        for epoch in range(epochs):
            for imgs, _ in source_loader:
                imgs = imgs.to('cuda')
                labels = torch.zeros(imgs.shape[0], dtype=torch.long, device='cuda')
                logits = self(imgs, None)
                loss = criterion(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            for imgs, _ in target_loader:
                imgs = imgs.to('cuda')
                labels = torch.ones(imgs.shape[0], dtype=torch.long, device='cuda')
                logits = self(imgs, None)
                loss = criterion(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        self.eval()