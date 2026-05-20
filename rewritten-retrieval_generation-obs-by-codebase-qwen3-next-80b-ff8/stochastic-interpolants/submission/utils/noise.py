import torch
import numpy as np

class NoiseSchedule:
    def __init__(self, timesteps=1000, schedule='linear'):
        self.timesteps = timesteps
        self.schedule = schedule
        self.beta = self._get_beta_schedule()
        
    def _get_beta_schedule(self):
        if self.schedule == 'linear':
            return torch.linspace(0.0001, 0.02, self.timesteps)
        elif self.schedule == 'cosine':
            s = 0.008
            steps = self.timesteps + 1
            x = torch.linspace(0, self.timesteps, steps)
            alphas_cumprod = torch.cos(((x / self.timesteps) + s) / (1 + s) * torch.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clip(betas, 0.0001, 0.999)
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")
    
    def sample_noise(self, shape, t, device):
        """Sample noise at time t"""
        noise = torch.randn(shape, device=device)
        return noise
    
    def get_alpha_t(self, t):
        """Get alpha_t = product of (1 - beta_s) for s=1 to t"""
        return torch.prod(1 - self.beta[:t+1]) if t > 0 else 1.0
    
    def get_sigma_t(self, t):
        """Get sigma_t for the noise schedule"""
        return torch.sqrt(1 - self.get_alpha_t(t))

def corrupt_image(image, mask, noise_level=0.1):
    """Corrupt image according to data-dependent coupling: x₀ = x₁ ⊕ (1 - mask) ⊙ ε"""
    noise = torch.randn_like(image) * noise_level
    corrupted = image * mask + noise * (1 - mask)
    return corrupted

def sample_base_density(target_samples, mask, noise_level=0.1):
    """Sample from base density using data-dependent coupling"""
    return corrupt_image(target_samples, mask, noise_level)