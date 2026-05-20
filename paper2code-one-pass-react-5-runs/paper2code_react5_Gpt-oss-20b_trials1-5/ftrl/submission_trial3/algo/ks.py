import torch
import numpy as np
from torch.distributions import Categorical

class KS:
    """
    Kick‑starting wrapper.
    Computes KL(π_teacher || π_student) on online data.
    """
    def __init__(self, teacher_policy, weight=0.5, decay=0.99998):
        self.teacher = teacher_policy
        self.weight = weight
        self.decay = decay
        # Freeze teacher
        for p in self.teacher.parameters():
            p.requires_grad = False

    def loss(self, policy, batch_obs):
        with torch.no_grad():
            teacher_logits = self.teacher(batch_obs)
            teacher_probs = torch.softmax(teacher_logits, dim=-1)
        logits = policy(batch_obs)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        kl = torch.sum(teacher_probs * (torch.log(teacher_probs + 1e-10) - log_probs), dim=-1).mean()
        loss = self.weight * kl
        # Decay weight
        self.weight *= self.decay
        return loss