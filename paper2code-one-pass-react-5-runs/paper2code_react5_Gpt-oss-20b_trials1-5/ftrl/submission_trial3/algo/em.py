import torch
import numpy as np
from torch.distributions import Categorical

class EM:
    """
    Episodic Memory wrapper.
    Keeps a fixed‑size buffer of pre‑trained transitions and samples from it
    to compute BC loss during fine‑tuning.
    """
    def __init__(self, teacher_policy, buffer, batch_size=64, weight=1.0):
        """
        teacher_policy: nn.Module frozen.
        buffer: list of (obs, action, reward, next_obs, done) tuples from pre‑training.
        """
        self.teacher = teacher_policy
        self.buffer = buffer
        self.batch_size = batch_size
        self.weight = weight
        for p in self.teacher.parameters():
            p.requires_grad = False

    def loss(self, policy):
        if len(self.buffer) < self.batch_size:
            return 0.0
        idx = np.random.choice(len(self.buffer), self.batch_size, replace=False)
        obs_batch = [self.buffer[i][0] for i in idx]
        obs_tensor = torch.tensor(obs_batch, dtype=torch.float32).to(policy.device)
        with torch.no_grad():
            teacher_logits = self.teacher(obs_tensor)
            teacher_probs = torch.softmax(teacher_logits, dim=-1)
        logits = policy(obs_tensor)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        kl = torch.sum(teacher_probs * (torch.log(teacher_probs + 1e-10) - log_probs), dim=-1).mean()
        return self.weight * kl