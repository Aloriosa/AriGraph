# utils.py
import torch
import math

# Simple VESDE: forward noise x_t = x0 + sigma_t * eps
def sigma_t(t: torch.Tensor, sigma_min: float = 0.0001, sigma_max: float = 15.0) -> torch.Tensor:
    """Schedule for the variance exploding SDE."""
    return sigma_min * (sigma_max / sigma_min) ** t

def sample_forward(x0: torch.Tensor, t: torch.Tensor, device: torch.device):
    """Generate noisy sample x_t given x0 and noise level t."""
    sigma = sigma_t(t.to(x0.dtype), sigma_min=0.0001, sigma_max=15.0).to(device)
    eps = torch.randn_like(x0)
    return x0 + sigma * eps, eps, sigma

def target_score(x0: torch.Tensor, x_t: torch.Tensor, sigma: torch.Tensor):
    """Analytical score of the forward SDE: (x0 - x_t) / sigma^2."""
    return (x0 - x_t) / (sigma ** 2)

def reverse_step(
    model: nn.Module,
    x_t: torch.Tensor,
    t: torch.Tensor,
    dt: float,
    cond_mask: torch.Tensor,
    cond_values: torch.Tensor | None,
    tokenizer: nn.Module,
    device: torch.device,
):
    """
    One reverse diffusion step (Euler–Maruyama).

    Parameters:
        model: Simformer
        x_t: current noisy sample (B,N,embed_dim)
        t: current time scalar tensor of shape (B,1)
        dt: time step size
        cond_mask: (B,N) bool mask of conditioned tokens
        cond_values: observed values for conditioned tokens (B,N,embed_dim)
    """
    # Build tokens for current x_t
    # Use placeholder identifiers and condition state = 0 (latent)
    B, N, D = x_t.shape
    ids = torch.zeros((B, N), dtype=torch.long, device=device)
    cond_state = torch.zeros((B, N), dtype=torch.long, device=device)
    tokens = tokenizer(ids, x_t, cond_state)

    # Compute score
    scores = model(tokens)  # (B,N,D)

    # Guidance: keep conditioned tokens fixed
    if cond_values is not None:
        scores = scores * (~cond_mask).unsqueeze(-1) + 0.0 * cond_mask.unsqueeze(-1)

    # SDE coefficients for VESDE
    sigma = sigma_t(t, sigma_min=0.0001, sigma_max=15.0).to(device)
    # Reverse drift
    drift = - (sigma ** 2) * scores
    # Sample noise
    eps = torch.randn_like(x_t)
    x_next = x_t + drift * dt + sigma * torch.sqrt(dt) * eps

    # Replace conditioned tokens with observed values
    if cond_values is not None:
        x_next = torch.where(cond_mask.unsqueeze(-1), cond_values, x_next)

    return x_next