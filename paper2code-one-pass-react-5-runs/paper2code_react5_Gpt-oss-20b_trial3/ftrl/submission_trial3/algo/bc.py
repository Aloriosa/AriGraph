import torch
import numpy as np
from torch.distributions import Categorical

class BC:
    """
    Behavioural Cloning wrapper.
    Computes KL(π_teacher || π_student) on a buffer of observations.
    """
    def __init__(self, teacher_policy, buffer, weight=2.0):
        """
        teacher_policy: nn.Module that outputs logits, frozen.
        buffer: list of (obs, action) tuples from pre‑training.
        weight: scalar weight for the KL loss.
        """
        self.teacher = teacher_policy
        self.buffer = buffer  # list of (obs, action)
        self.weight = weight
        # Freeze teacher
        for p in self.teacher.parameters():
            p.requires_grad = False

    def loss(self, policy, batch_obs):
        """
        batch_obs: torch tensor of shape (N, obs_dim)
        Returns weighted KL loss.
        """
        with torch.no_grad():
            teacher_logits = self.teacher(batch_obs)
            teacher_probs = torch.softmax(teacher_logits, dim=-1)
        logits = policy(batch_obs)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        # KL(teacher || student)
        kl = torch.sum(teacher_probs * (torch.log(teacher_probs + 1e-10) - log_probs), dim=-1).mean()
        return self.weight * kl