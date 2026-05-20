"""
Elastic Weight Consolidation loss.
Computes a diagonal Fisher information matrix on the teacher policy
and uses it to penalise changes of the student parameters.
"""
import torch
import torch.nn.functional as F

def compute_fisher(teacher_policy, data_loader):
    """
    Estimate diagonal Fisher using a small batch of data.
    data_loader yields tensors of shape [batch, obs_dim].
    """
    fisher = {}
    for param in teacher_policy.parameters():
        fisher[param] = torch.zeros_like(param, device=param.device)

    teacher_policy.eval()
    for batch in data_loader:
        batch = batch.to(next(teacher_policy.parameters()).device)
        probs = teacher_policy(batch).squeeze()
        logits = torch.log(probs + 1e-12) - torch.log(1 - probs + 1e-12)
        # for binary action, gradient of log p wrt logits is (1-p) for action 1
        # but we approximate Fisher as squared gradient of log prob
        grads = torch.autograd.grad(probs.sum(), teacher_policy.parameters(), create_graph=True)
        for p, g in zip(teacher_policy.parameters(), grads):
            fisher[p] += g.pow(2)
    # average over samples
    for p in fisher:
        fisher[p] /= len(data_loader)
    return fisher

def ewc_loss(student_policy, fisher, theta_star, lambda_ewc=1e3):
    loss = 0.0
    for p, f, p_star in zip(student_policy.parameters(),
                            fisher.values(),
                            theta_star):
        loss += (f * (p - p_star).pow(2)).sum()
    return lambda_ewc * loss