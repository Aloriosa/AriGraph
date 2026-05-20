import torch
import torch.nn.functional as F

def kl_divergence(p_logits, q_logits):
    """
    KL divergence between two Bernoulli distributions given logits.
    D_KL(p||q) = p*log(p/q) + (1-p)*log((1-p)/(1-q))
    """
    p = torch.sigmoid(p_logits)
    q = torch.sigmoid(q_logits)
    # Avoid log(0) by adding epsilon
    eps = 1e-12
    p = torch.clamp(p, eps, 1 - eps)
    q = torch.clamp(q, eps, 1 - eps)
    kl = p * torch.log(p / q) + (1 - p) * torch.log((1 - p) / (1 - q))
    return kl.mean()

def fisher_information(policy, env, num_samples=100):
    """
    Approximate diagonal Fisher information matrix for policy parameters.
    For the simple policy we return a dict mapping param name to a tensor.
    """
    device = next(policy.parameters()).device
    fisher = {name: torch.zeros_like(param, device=device) for name, param in policy.named_parameters()}
    policy.eval()
    for _ in range(num_samples):
        state = torch.tensor(env.reset(), dtype=torch.float32).unsqueeze(0).to(device)
        logits = policy(state)
        # Sample action
        action_dist = torch.distributions.Bernoulli(logits=torch.sigmoid(logits))
        action = action_dist.sample()
        # Compute log prob
        log_prob = action_dist.log_prob(action)
        # Compute gradients
        policy.zero_grad()
        log_prob.backward()
        for name, param in policy.named_parameters():
            fisher[name] += param.grad.detach() ** 2
    for name in fisher:
        fisher[name] /= num_samples
    policy.train()
    return fisher