import torch
import torch.nn.functional as F

class SDESolver:
    def __init__(self, velocity_model, noise_schedule, timesteps=1000):
        self.velocity_model = velocity_model
        self.noise_schedule = noise_schedule
        self.timesteps = timesteps
        
    def sample(self, shape, device, mask=None, guidance_scale=1.0, verbose=False):
        """Generate samples using the learned velocity field"""
        x = torch.randn(shape, device=device)
        
        # Create time steps
        timesteps = torch.linspace(1, 0, self.timesteps + 1, device=device)
        
        if verbose:
            from tqdm import tqdm
            timesteps = tqdm(timesteps[:-1])
        
        for i, t in enumerate(timesteps[:-1]):
            t_batch = torch.full((shape[0],), t, device=device)
            
            # Predict velocity
            with torch.no_grad():
                velocity = self.velocity_model(x, t_batch)
            
            # Compute dt
            dt = timesteps[i] - timesteps[i + 1]
            
            # Euler-Maruyama step
            x = x + velocity * dt
            
            # Add noise if we're not at the final step
            if i < len(timesteps) - 2:
                noise = torch.randn_like(x)
                sigma_t = self.noise_schedule.get_sigma_t(int(t * self.timesteps))
                x = x + noise * sigma_t * torch.sqrt(dt)
        
        return x
    
    def sample_with_mask(self, shape, device, mask, guidance_scale=1.0, verbose=False):
        """Generate samples with conditional masking for inpainting"""
        # Start with corrupted image
        x = torch.randn(shape, device=device)
        
        # Create time steps
        timesteps = torch.linspace(1, 0, self.timesteps + 1, device=device)
        
        if verbose:
            from tqdm import tqdm
            timesteps = tqdm(timesteps[:-1])
        
        for i, t in enumerate(timesteps[:-1]):
            t_batch = torch.full((shape[0],), t, device=device)
            
            # Predict velocity
            with torch.no_grad():
                velocity = self.velocity_model(x, t_batch)
            
            # Compute dt
            dt = timesteps[i] - timesteps[i + 1]
            
            # Euler-Maruyama step
            x = x + velocity * dt
            
            # Add noise if we're not at the final step
            if i < len(timesteps) - 2:
                noise = torch.randn_like(x)
                sigma_t = self.noise_schedule.get_sigma_t(int(t * self.timesteps))
                x = x + noise * sigma_t * torch.sqrt(dt)
            
            # Apply mask conditioning
            if mask is not None:
                # Keep the original masked region unchanged
                x = x * (1 - mask) + mask * x  # This is a placeholder - in practice we'd preserve the original data
        
        return x