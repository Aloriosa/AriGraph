import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearPolicy(nn.Module):
    """
    Simple linear policy: pi(a=1 | s) = sigmoid(w * s + b)
    """
    def __init__(self, input_dim=1, output_dim=1):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=True)

    def forward(self, x):
        logits = self.linear(x)
        probs = torch.sigmoid(logits)
        return probs

    def get_action(self, state):
        """
        state: torch.Tensor of shape (1, input_dim)
        Returns: action (0 or 1)
        """
        with torch.no_grad():
            probs = self.forward(state)
            return torch.bernoulli(probs).item()