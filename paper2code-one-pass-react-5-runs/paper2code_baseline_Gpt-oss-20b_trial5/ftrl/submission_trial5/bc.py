"""
Behavioral cloning helper: compute MSE loss between student and teacher action probabilities
on a collection of pre‑trained states.
"""
import torch


def bc_loss(student: LinearPolicy, teacher: LinearPolicy, buffer_states: torch.Tensor):
    """
    :param student: current policy
    :param teacher: pre‑trained teacher policy
    :param buffer_states: tensor of shape [N, 2]
    :return: scalar loss
    """
    with torch.no_grad():
        teacher_probs = teacher.forward(buffer_states).detach()  # [N, 2]
    student_probs = student.forward(buffer_states)  # [N, 2]
    loss = F.mse_loss(student_probs, teacher_probs)
    return loss