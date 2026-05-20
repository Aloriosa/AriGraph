import torch
import torch.nn as nn

class PolicyModule(nn.Module):
    """
    A very small policy module used inside CompoNet.
    For simplicity we use a 2‑layer MLP with ReLU activations.
    """
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        hidden = 128
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def act(self, obs: torch.Tensor) -> int:
        with torch.no_grad():
            logits = self.forward(obs)
            probs = torch.softmax(logits, dim=0)
            return int(torch.multinomial(probs, 1).item())

class CompoNet(nn.Module):
    """
    Minimal CompoNet implementation.
    Maintains a list of policy modules; after training on a task,
    the current module is frozen and a new trainable module is added.
    """
    def __init__(self, obs_dim: int = 512, action_dim: int = 6):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.modules = nn.ModuleList()
        self.add_module()  # start with one module

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def add_module(self):
        module = PolicyModule(self.obs_dim, self.action_dim)
        self.modules.append(module)

    def freeze_current(self):
        """Freeze the last added module."""
        if self.modules:
            for param in self.modules[-1].parameters():
                param.requires_grad = False

    def act(self, obs: torch.Tensor) -> int:
        """
        Forward pass through the most recent module only.
        In a full implementation this would compose over all modules.
        """
        return self.modules[-1].act(obs)

    def train_step(self, loss: torch.Tensor):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()