import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearPolicy(nn.Module):
    """
    A simple linear policy mapping the one‑hot encoded state to
    a probability of taking action 1.  The policy outputs a
    Bernoulli distribution over actions.
    """

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)  # input: one‑hot state, output: logit

    def forward(self, state):
        """
        Parameters
        ----------
        state : torch.Tensor of shape (batch, 1)
            Integer state indices (0 or 1).

        Returns
        -------
        probs : torch.Tensor of shape (batch, 2)
            Action probabilities for actions 0 and 1.
        """
        state_onehot = F.one_hot(state.squeeze(-1), num_classes=2).float()
        logits = self.linear(state_onehot)
        probs = torch.sigmoid(logits).squeeze(-1)
        return probs

    def sample(self, state):
        probs = self.forward(state)
        dist = torch.distributions.Bernoulli(probs)
        action = dist.sample()
        logp = dist.log_prob(action)
        return action.long(), logp