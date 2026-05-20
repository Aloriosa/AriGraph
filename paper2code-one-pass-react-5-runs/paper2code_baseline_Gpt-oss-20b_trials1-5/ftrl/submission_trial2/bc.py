"""
Behavioral Cloning loss: KL divergence between teacher and student policies
evaluated on a fixed buffer of teacher states.
"""
import torch
import torch.nn.functional as F

def bc_loss(student_policy, teacher_policy, buffer_states):
    """
    buffer_states: Tensor of shape [N, obs_dim]
    """
    student_probs = student_policy(buffer_states).squeeze()
    with torch.no_grad():
        teacher_probs = teacher_policy(buffer_states).squeeze()
    # KL divergence between categorical distributions with 2 actions
    # p = teacher_probs, q = student_probs
    # D_KL(p||q) = p*log(p/q) + (1-p)*log((1-p)/(1-q))
    eps = 1e-12
    p = teacher_probs.clamp(min=eps, max=1-eps)
    q = student_probs.clamp(min=eps, max=1-eps)
    kl = p * torch.log(p / q) + (1 - p) * torch.log((1 - p) / (1 - q))
    return kl.mean()