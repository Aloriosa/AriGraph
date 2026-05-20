import torch
import torch.nn.functional as F


def compute_gae(rewards, values, dones, gamma=0.99, tau=0.95):
    """
    Compute Generalized Advantage Estimation (GAE) and returns.
    """
    advantages = torch.zeros_like(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * tau * (1 - dones[t]) * gae
        advantages[t] = gae
    returns = advantages + values[:-1]
    return advantages.detach(), returns.detach()


def flatten_dict(dict_of_lists):
    """
    Flatten a list of dicts into a single dict of concatenated tensors.
    """
    flat = {}
    for key in dict_of_lists[0].keys():
        flat[key] = torch.cat([d[key] for d in dict_of_lists], dim=0)
    return flat