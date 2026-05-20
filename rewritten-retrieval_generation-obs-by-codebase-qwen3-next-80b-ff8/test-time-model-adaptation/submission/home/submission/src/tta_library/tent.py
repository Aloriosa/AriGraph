import torch
import torch.nn as nn
from copy import deepcopy
from utils.cli_utils import entropy_loss

class Tent(nn.Module):
    """Tent adapts a model by entropy minimization during testing.
    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)
        self.entropy_loss = entropy_loss()

    def forward(self, x, y=None):
        if self.episodic:
            self.reset()
        if self.steps > 0:
            for _ in range(self.steps):
                outputs = forward_and_adapt(x, self.model, self.optimizer, self.entropy_loss)
        else:
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(x)
        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)

    def reset_steps(self, new_steps):
        self.steps = new_steps

def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state

def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Load the model and optimizer states."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)

def forward_and_adapt(x, model, optimizer, entropy_loss):
    """Forward and adapt model on batch of data."""
    model.train()
    outputs = model(x)
    loss = entropy_loss(outputs)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return outputs