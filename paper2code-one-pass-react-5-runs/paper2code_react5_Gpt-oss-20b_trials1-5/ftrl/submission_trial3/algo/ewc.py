import torch
import numpy as np

class EWC:
    """
    Elastic Weight Consolidation wrapper.
    Computes Fisher on a data loader of (obs, action) pairs.
    Applies penalty λ * sum_i F_i * (θ_i - θ*_i)^2 on policy parameters.
    """
    def __init__(self, policy, lambda_=200.0):
        self.policy = policy
        self.lambda_ = lambda_
        self.fisher = {}
        self.theta_pre = {}

    def compute_fisher(self, data_loader):
        # Initialize fisher dict
        for name, param in self.policy.named_parameters():
            self.fisher[name] = torch.zeros_like(param.data)

        self.policy.eval()
        total_samples = 0
        for batch in data_loader:
            batch_obs, batch_act = batch
            batch_obs = batch_obs.to(self.policy.device)
            batch_act = batch_act.to(self.policy.device)
            logits = self.policy(batch_obs)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            # For each sample compute grad of log prob of its taken action
            for i in range(batch_obs.size(0)):
                self.policy.zero_grad()
                log_prob = log_probs[i, batch_act[i]]
                log_prob.backward(retain_graph=True)
                for name, param in self.policy.named_parameters():
                    if param.grad is not None:
                        self.fisher[name] += param.grad.data.clone() ** 2
            total_samples += batch_obs.size(0)

        # Normalise
        for name in self.fisher:
            self.fisher[name] /= total_samples
        # Store pre‑trained parameters
        self.theta_pre = {name: param.clone() for name, param in self.policy.named_parameters()}

    def loss(self):
        penalty = 0.0
        for name, param in self.policy.named_parameters():
            if name in self.fisher:
                penalty += (self.fisher[name] * (param - self.theta_pre[name]) ** 2).sum()
        return self.lambda_ * penalty