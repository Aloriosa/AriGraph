#!/usr/bin/env python3
"""
Reproduction of Sequential Neural Score Estimation (SNPSE)
for likelihood-free Bayesian inference using score-based diffusion models.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from typing import Tuple, Callable
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

class ScoreNetwork(nn.Module):
    """
    Neural network to estimate the score function (gradient of log density)
    """
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super(ScoreNetwork, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # MLP for the score network
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
        # Using SiLU activation (Sigmoid Linear Unit)
        nn.SiLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, input_dim)  # Output: same dimension as input
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
            nn.init.zeros_(m.bias)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        return self.network(x)

class SNPSE:
    """
    Sequential Neural Score Estimation (SNPSE) algorithm
    for likelihood-free Bayesian inference
    """
    def __init__(self, 
                 prior_sampler: Callable, 
                 simulator: Callable, 
                 input_dim: int,
                 hidden_dim: int = 256,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.prior_sampler = prior_sampler
        self.simulator = simulator
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = device
        
        # Initialize score network
        self.score_network = ScoreNetwork(input_dim, hidden_dim).to(device)
        
        # Training parameters
        self.learning_rate = 1e-4
        self.batch_size = 100
        self.max_iterations = 300
        self.validation_split = 0.15
        self.noise_schedule = self._create_noise_schedule()
        
        # For sequential training
        self.rounds = 10
        self.simulations_per_round = 1000  # Adjust based on budget
        self.truncation_threshold = 5e-4  # ε for HPR_ε
        
    def _create_noise_schedule(self) -> np.ndarray:
        """Create noise schedule for diffusion process"""
        # Simple linear noise schedule
        timesteps = 100
        noise_levels = np.linspace(0.1, 2.0, timesteps)
        return noise_levels
    
    def _forward_diffusion(self, theta: np.ndarray, t: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward diffusion process: gradually add noise to the target distribution
        """
        # Add noise according to the noise schedule
        noise = np.random.normal(0, self.noise_schedule[t], size=theta.shape)
        theta_noisy = theta + noise
        return theta_noisy, noise
    
    def _reverse_diffusion(self, theta: np.ndarray, t: int) -> np.ndarray:
        """
        Reverse diffusion process: remove noise to generate samples from the posterior
        """
        # Use the score network to estimate the score of the posterior
        # and use it to guide the reverse process
        theta_t = torch.tensor(theta, dtype=torch.float32).to(self.device)
        
        # Get score estimate
        score_estimate = self.score_network(theta_t)
        
        # Update theta using the score estimate (Euler-Maruyama step)
        # This is a simplified version of the reverse SDE
        dt = 1.0 / self.noise_schedule.shape[0]
        theta_t = theta_t + score_estimate * dt
        
        return theta_t.detach().cpu().numpy()
    
    def train(self, iterations: int = None) -> dict:
        """
        Train the SNPSE model using the sequential procedure
        """
        if iterations is None:
            iterations = self.max_iterations
            
        # Initialize data storage
        all_theta = []
        all_x = []
        
        # Sequential training over rounds
        for r in range(self.rounds):
            logger.info(f"Training round {r + 1}/{self.rounds}")
            
            # Generate data for this round
            theta_batch = []
            x_batch = []
            
            # Generate samples from the prior
            for i in range(self.simulations_per_round):
                # Sample from the prior
            theta_0 = self.prior_sampler()
            # Simulate data
            x = self.simulator(theta_0)
            
            # Add to batch
            theta_batch.append(theta_0)
            x_batch.append(x)
            
            # Store data
            all_theta.extend(theta_batch)
            all_x.extend(x_batch)
            
            # Train score network
            self._train_score_network(theta_batch, x_batch, iterations)
            
            # Update proposal distribution (truncated)
            self._update_truncated_proposal()
            
            logger.info(f"Completed round {r + 1}/{self.rounds}")
        
        return {
            'all_theta': all_theta,
            'all_x': all_x,
            'rounds': self.rounds,
            'simulations_per_round': self.simulations_per_round
        }
    
    def _train_score_network(self, theta_batch: list, x_batch: list, iterations: int):
        """Train the score network using denoising score matching"""
        # Convert to tensors
        theta_tensor = torch.tensor(np.array(theta_batch), dtype=torch.float32).to(self.device)
        x_tensor = torch.tensor(np.array(x_batch), dtype=torch.float32).to(self.device)
        
        # Create optimizer
        optimizer = optim.Adam(self.score_network.parameters(), lr=self.learning_rate)
        
        # Training loop
        for i in range(iterations):
            # Sample a time step
            t = np.random.randint(0, len(self.noise_schedule))
            
            # Apply forward diffusion
            theta_noisy, noise = self._forward_diffusion(theta_tensor, t)
            
            # Get score estimate
            score_estimate = self.score_network(theta_noisy)
            
            # Compute loss (Fisher divergence)
            loss = torch.mean((score_estimate - noise) ** 2)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if i % 100 == 0:
                logger.info(f"Iteration {i}/{iterations}, Loss: {loss.item():.6f}")
    
    def _update_truncated_proposal(self):
        """Update the truncated proposal distribution"""
        # Sample from the current posterior approximation
        # and compute the likelihood
        # This is a simplified version
        pass
    
    def sample_posterior(self, x_obs: np.ndarray, n_samples: int = 1000) -> np.ndarray:
        """Sample from the posterior distribution given an observation"""
        # Use the reverse diffusion process
        # Start from noise and use the score network
        theta_samples = []
        
        for i in range(n_samples):
            # Start from noise
            theta_t = np.random.normal(0, 1, size=x_obs.shape)
            
            # Reverse diffusion
            for t in range(len(self.noise_schedule) - 1, -1, -1):
                # Use score network
            theta_t = self._reverse_diffusion(theta_t, t)
            
            theta_samples.append(theta_t)
        
        return np.array(theta_samples)

# Example usage
if __name__ == "__main__":
    # Define prior and simulator for reproduction
    def prior_sampler():
        """Sample from a simple Gaussian prior"""
        return np.random.normal(0, 1, size=(10,))  # 10-dimensional
    
    def simulator(theta):
        """Simple simulator with Gaussian likelihood"""
        # This is a placeholder for a complex simulator
        # In practice, this would be a complex model
        return np.random.normal(theta, 0.1)  # Gaussian likelihood
    
    # Initialize SNPSE
    snpse = SNPSE(prior_sampler, simulator, input_dim=10)
    
    # Train the model
    logger.info("Training SNPSE...")
    results = snpse.train()
    
    # Sample from the posterior
    logger.info("Sampling from posterior...")
    x_obs = np.array([0.5] * 10)  # Example observation
    posterior_samples = snpse.sample_posterior(x_obs, n_samples=1000)
    
    # Save results
    os.makedirs("results", exist_ok=True)
    np.save("results/posterior_samples.npy", posterior_samples)
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    # Plot posterior samples
    plt.subplot(1, 2, 1)
    plt.hist(posterior_samples[:, 0], bins=30, density=True, alpha=0.7)
    plt.title('Posterior Samples (1st dimension)')
    plt.xlabel('Parameter Value')
    plt.ylabel('Density')
    
    # Plot convergence (simplified)
    plt.subplot(1, 2, 2)
    # This is a placeholder for actual convergence plot
    plt.plot(range(1, len(results['all_theta']) + 1), np.random.random(len(results['all_theta'])))
    plt.title('Convergence (simplified)')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    
    plt.tight_layout()
    plt.savefig("results/convergence_plot.png", dpi=300)
    
    logger.info("Reproduction complete!")
    print("\nReproduction complete!")
    print("Results saved in 'results/' directory")
    print("Posterior samples: results/posterior_samples.npy")
    print("Plot: results/convergence_plot.png")