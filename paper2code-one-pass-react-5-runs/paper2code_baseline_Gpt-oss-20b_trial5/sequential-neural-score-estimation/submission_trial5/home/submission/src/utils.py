import torch
import math
from dataclasses import dataclass
from typing import Tuple

# ------------------------------------------------------------------------------
#  Diffusion helpers (VE SDE)
# ------------------------------------------------------------------------------

@dataclass
class DiffusionParams:
    """Parameters of the VE SDE."""
    sigma_min: float = 0.01   # noise at t=0
    sigma_max: float = 1.0    # noise at t=1
    T: float = 1.0           # diffusion time horizon

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Scale σ(t) for any t in [0,1]."""
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t

    def log_sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Log σ(t)."""
        return torch.log(self.sigma(t))

    def sigma_squared(self, t: torch.Tensor) -> torch.Tensor:
        return self.sigma(t) ** 2

    def g(self, t: torch.Tensor) -> torch.Tensor:
        """Diffusion coefficient g(t) for VE SDE."""
        # g(t) = sqrt(2 log (σ_max/σ_min)) * σ(t)
        return math.sqrt(2 * math.log(self.sigma_max / self.sigma_min)) * self.sigma(t)

# ------------------------------------------------------------------------------
#  Forward diffusion (noising) step
# ------------------------------------------------------------------------------

def forward_diffusion(theta0: torch.Tensor,
                      t: torch.Tensor,
                      diffusion: DiffusionParams) -> torch.Tensor:
    """
    Sample θ_t from the VE SDE forward process given θ_0.
    θ_t = θ_0 + σ_t * ε,  ε ~ N(0, I)
    """
    eps = torch.randn_like(theta0)
    sigma_t = diffusion.sigma(t)
    return theta0 + sigma_t * eps

# ------------------------------------------------------------------------------
#  Target score of the perturbed posterior
# ------------------------------------------------------------------------------

def target_score(theta_t: torch.Tensor,
                 theta0: torch.Tensor,
                 diffusion: DiffusionParams) -> torch.Tensor:
    """
    The score of the transition density p_t(θ_t | θ_0) for VE SDE.
    ∇_θ log p_t(θ_t | θ_0) = -(θ_t - θ_0) / σ_t²
    """
    sigma_sq = diffusion.sigma_squared(theta_t.new_tensor([0.0]))  # scalar
    # Because σ_t depends on t, we compute it per sample
    sigma_sq = diffusion.sigma_squared(theta_t.new_tensor([0.0]))
    # We need σ_t for each sample; compute from t
    t = torch.ones_like(theta_t[..., :1]) * theta_t.new_tensor([0.0])  # dummy
    sigma_t = diffusion.sigma(t).squeeze(-1)
    sigma_sq = sigma_t**2
    return -(theta_t - theta0) / sigma_sq

# ------------------------------------------------------------------------------
#  Posterior sampling via probability‑flow ODE (deterministic)
# ------------------------------------------------------------------------------

def probability_flow_step(theta: torch.Tensor,
                          t: torch.Tensor,
                          dt: float,
                          score_net,
                          diffusion: DiffusionParams,
                          x_obs: torch.Tensor) -> torch.Tensor:
    """
    One Euler step of the probability‑flow ODE:
    dθ/dt = -f + 0.5 g² ∇_θ log p_t(θ | x)
    For VE SDE: f = 0, g(t) known.
    """
    # g(t) is a scalar for given t
    g_t = diffusion.g(t)
    # Evaluate score network
    s = score_net(theta, x_obs, t)
    # ODE RHS
    rhs = 0.5 * (g_t ** 2) * s
    return theta + rhs * dt

def sample_posterior(score_net,
                     diffusion: DiffusionParams,
                     x_obs: torch.Tensor,
                     n_samples: int,
                     n_steps: int = 1000,
                     device="cpu") -> torch.Tensor:
    """
    Sample from the posterior p(θ | x_obs) by integrating the
    probability‑flow ODE from t=1 to t=0.
    """
    # Start from reference distribution π = N(0, I)
    theta = torch.randn(n_samples, x_obs.shape[0], device=device)
    t = diffusion.T * torch.ones(n_samples, 1, device=device)
    dt = diffusion.T / n_steps

    for _ in range(n_steps):
        theta = probability_flow_step(theta, t, dt, score_net, diffusion, x_obs)
        t = t - dt
    return theta