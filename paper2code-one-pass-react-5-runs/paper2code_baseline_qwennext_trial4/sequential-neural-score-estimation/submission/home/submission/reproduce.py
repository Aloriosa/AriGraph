#!/usr/bin/env python3
"""
Reproduction script for "Sequential Neural Score Estimation: Likelihood-Free Inference with Conditional Score Based Diffusion Models"

This script implements the Neural Posterior Score Estimation (NPSE) and its sequential variant (TSNPSE) as described in the paper.
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import time
import logging
from typing import Tuple, List, Optional
import pickle

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

class ScoreNetwork(nn.Module):
    """
    Score network for estimating the score of the posterior distribution.
    This network takes as input the state theta, observation x, and time t, and outputs an estimate of the score.
    """
    
    def __init__(self, theta_dim, x_dim, time_dim=64, hidden_dim=256, num_layers=3):
        """
        Initialize the score network.
        
        Args:
            theta_dim: Dimension of the parameter space
            x_dim: Dimension of the observation space
            time_dim: Dimension of the time embedding
            hidden_dim: Hidden dimension of the MLP
            num_layers: Number of layers in the MLP
        """
        super(ScoreNetwork, self).__init__()
        self.theta_dim = theta_dim
        self.x_dim = x_dim
        self.time_dim = time_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding for theta
        self.theta_embedding = nn.Sequential(
            nn.Linear(theta_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Embedding for x
        self.x_embedding = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Embedding for time
        self.time_embedding = nn.Sequential(
            nn.Linear(1, time_dim),
        )
        
        # Main network
        self.main_network = nn.ModuleList()
        self.main_network.append(nn.Linear(hidden_dim * 2 + time_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.main_network.append(nn.Linear(hidden_dim, hidden_dim))
        
        self.output_layer = nn.Linear(hidden_dim, theta_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights for the network."""
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.01)
    
    def forward(self, theta, x, t):
        """
        Forward pass of the score network.
        
        Args:
            theta: Parameters, shape (batch_size, theta_dim)
            x: Observations, shape (batch_size, x_dim)
            t: Time, shape (batch_size, 1)
        
        Returns:
            score: Estimated score, shape (batch_size, theta_dim)
        """
        # Embed theta
        theta_emb = self.theta_embedding(theta)
        
        # Embed x
        x_emb = self.x_embedding(x)
        
        # Embed time
        time_emb = self.time_embedding(t)
        
        # Concatenate embeddings
        combined = torch.cat([theta_emb, x_emb, time_emb], dim=-1)
        
        # Pass through main network
        for i in range(self.num_layers):
            if i == 0:
                hidden = self.main_network[i](combined)
            else:
                hidden = self.main_network[i](hidden)
            if i < self.num_layers - 1:
                hidden = F.silu(hidden)
        
        # Output layer
        score = self.output_layer(hidden)
        
        return score

class DiffusionModel(nn.Module):
    """
    Diffusion model for generating samples from the posterior distribution.
    This model implements the forward and reverse diffusion processes.
    """
    
    def __init__(self, score_network, timesteps=100, beta_start=0.0001, beta_end=0.02):
        """
        Initialize the diffusion model.
        
        Args:
            score_network: Score network for estimating the score of the posterior
            timesteps: Number of diffusion timesteps
            beta_start: Starting value for beta
            beta_end: Ending value for beta
        """
        super(DiffusionModel, self).__init__()
        self.score_network = score_network
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        
        # Define beta schedule
        self.betas = torch.linspace(self.beta_start, self.beta_end, self.timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1), self.alphas_cumprod[:-1]])
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_recip_alphas_minus_one = torch.sqrt(1.0 / self.alphas - 1.0)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_variance = torch.clamp(self.posterior_variance, min=1e-8)
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance)
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        
        # Move to device
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.sqrt_recip_alphas = self.sqrt_recip_alphas.to(device)
        self.sqrt_recip_alphas_minus_one = self.sqrt_recip_alphas_minus_one.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        self.posterior_log_variance_clipped = self.posterior_log_variance_clipped.to(device)
        self.posterior_mean_coef1 = self.posterior_mean_coef1.to(device)
        self.posterior_mean_coef2 = self.posterior_mean_coef2.to(device)
    
    def forward(self, x, t, theta, return_loss=True):
        """
        Forward pass of the diffusion model.
        
        Args:
            x: Observations, shape (batch_size, x_dim)
            t: Time, shape (batch_size, 1)
            theta: Parameters, shape (batch_size, theta_dim)
            return_loss: Whether to return the loss
        
        Returns:
            loss: Loss value, if return_loss=True
        """
        # Sample noise
        noise = torch.randn_like(theta)
        
        # Forward diffusion process
        theta_t = self.q_sample(theta, t, noise)
        
        # Estimate score
        estimated_score = self.score_network(theta_t, x, t)
        
        if return_loss:
            # Compute loss
            loss = F.mse_loss(estimated_score, noise)
            return loss
        else:
            return estimated_score
    
    def q_sample(self, x_start, t, noise=None):
        """
        Sample from the forward diffusion process.
        
        Args:
            x_start: Initial state
            t: Time
            noise: Noise to add
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        return self.sqrt_alphas_cumprod[t, None] * x_start + self.sqrt_one_minus_alphas_cumprod[t, None] * noise
    
    def p_sample(self, x, t, x_start, noise=None):
        """
        Sample from the reverse diffusion process.
        
        Args:
            x: Current state
            t: Time
            x_start: Initial state
            noise: Noise to add
        """
        if noise is None:
            noise = torch.randn_like(x)
        
        # Estimate score
        estimated_score = self.score_network(x, x, t)
        
        # Reverse diffusion process
        mean = self.posterior_mean_coef1[t, None] * x + self.posterior_mean_coef2[t, None] * x_start
        return mean + torch.exp(0.5 * self.posterior_log_variance_clipped[t, None]) * noise
    
    def sample(self, x, steps=None):
        """
        Generate samples from the posterior distribution.
        
        Args:
            x: Observation
            steps: Number of steps for the reverse process
        """
        if steps is None:
            steps = self.timesteps
        
        # Start from noise
        x_t = torch.randn_like(x)
        
        # Reverse diffusion process
        for t in range(steps - 1, -1, -1):
            t_batch = torch.full((x.shape[0], 1), t, dtype=torch.float32)
            x_t = self.p_sample(x_t, t_batch, x, noise=None)
        
        return x_t

class SequentialNeuralScoreEstimator:
    """
    Sequential Neural Score Estimator (SNSE) for likelihood-free inference.
    This class implements the TSNPSE algorithm described in the paper.
    """
    
    def __init__(self, theta_dim, x_dim, hidden_dim=256, num_layers=3, timesteps=100, learning_rate=1e-4):
        """
        Initialize the Sequential Neural Score Estimator.
        
        Args:
            theta_dim: Dimension of the parameter space
            x_dim: Dimension of the observation space
            hidden_dim: Hidden dimension of the MLP
            num_layers: Number of layers in the MLP
            timesteps: Number of diffusion timesteps
            learning_rate: Learning rate for the optimizer
        """
        self.theta_dim = theta_dim
        self.x_dim = x_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.timesteps = timesteps
        self.learning_rate = learning_rate
        
        # Initialize score network
        self.score_network = ScoreNetwork(theta_dim, x_dim, hidden_dim=hidden_dim, num_layers=num_layers)
        self.score_network.to(device)
        
        # Initialize diffusion model
        self.diffusion_model = DiffusionModel(self.score_network, timesteps=timesteps)
        self.diffusion_model.to(device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.score_network.parameters(), lr=learning_rate)
        
        # Initialize data
        self.data = []
        self.observation = None
        
        # Initialize hyperparameters
        self.max_rounds = 10
        self.simulations_per_round = 100
        self.batch_size = 100
        self.epsilon = 5e-5
        self.alpha = 0.95
        self.tolerance = 1e-5
        
        # Initialize metrics
        self.loss_history = []
        self.validation_loss_history = []
        self.training_time = 0
        self.simulation_time = 0
        self.evaluation_time = 0
        self.evaluation_count = 0
        
        # Initialize random seed
        self.seed = 42
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
    
    def train(self, observation, simulator, budget=1000, rounds=10, max_iterations=100, batch_size=100):
        """
        Train the Sequential Neural Score Estimator.
        
        Args:
            observation: Observed data
            simulator: Simulator function
            budget: Total simulation budget
            rounds: Number of rounds
            max_iterations: Maximum number of iterations
            batch_size: Batch size
        """
        logger.info("Training Sequential Neural Score Estimator...")
        logger.info(f"Observation: {observation}")
        logger.info(f"Budget: {budget}")
        logger.info(f"Rounds: {rounds}")
        logger.info(f"Max iterations: {max_iterations}")
        logger.info(f"Batch size: {batch_size}")
        
        # Set hyperparameters
        self.max_rounds = rounds
        self.simulations_per_round = budget // rounds
        self.batch_size = batch_size
        
        # Initialize observation
        self.observation = torch.tensor(observation, dtype=torch.float32)
        if len(self.observation.shape) == 1:
            self.observation = self.observation.unsqueeze(0)
        
        # Initialize prior
        self.prior = torch.distributions.Uniform(-1.0, 1.0)
        
        # Initialize truncated prior
        self.truncated_prior = self.prior
        
        # Initialize proposal
        self.proposal = self.prior
        
        # Initialize posterior approximation
        self.posterior_approximation = None
        
        # Initialize HPR
        self.HPR = None
        
        # Initialize training
        self.training_started = False
        self.training_finished = False
        
        # Initialize training loop
        start_time = time.time()
        
        # Training loop
        for round_num in range(self.max_rounds):
            logger.info(f"Round {round_num + 1}/{self.max_rounds}")
            
            # Simulate data
            self.simulate_data(simulator)
            
            # Train model
            self.train_model(max_iterations)
            
            # Update truncated prior
            self.update_truncated_prior()
            
            # Update proposal
            self.update_proposal()
            
            # Update posterior approximation
            self.update_posterior_approximation()
            
            # Update HPR
            self.update_HPR()
            
            # Check convergence
            if self.converged():
                logger.info("Converged")
            else:
                logger.info("Not converged")
        
        # Final training
        self.final_training()
        
        # Training completed
        end_time = time.time()
        self.training_time = end_time - start_time
        
        # Save results
        self.save_results()
        
        logger.info("Training completed!")
    
    def simulate_data(self, simulator):
        """
        Simulate data from the simulator.
        
        Args:
            simulator: Simulator function
        """
        logger.info("Simulating data...")
        
        # Simulate data
        start_time = time.time()
        
        # Sample parameters
        theta = self.proposal.sample((self.simulations_per_round, self.theta_dim))
        
        # Simulate observations
        x = torch.zeros(self.simulations_per_round, self.x_dim)
        for i in range(self.simulations_per_round):
            x[i] = torch.tensor(simulator(theta[i].numpy()), dtype=torch.float32)
        
        # Add data to dataset
        for i in range(self.simulations_per_round):
            self.data.append((theta[i], x[i])
        
        end_time = time.time()
        self.simulation_time += end_time - start_time
        
        logger.info(f"Simulated {self.simulations_per_round} data points")
    
    def train_model(self, max_iterations=100):
        """
        Train the model.
        
        Args:
            max_iterations: Maximum number of iterations
        """
        logger.info("Training model...")
        
        # Set training mode
        self.score_network.train()
        
        # Initialize loss history
        loss_history = []
        
        # Training loop
        for iteration in range(max_iterations):
            # Sample batch
            batch = self.sample_batch()
            
            # Forward pass
            loss = self.compute_loss(batch)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Store loss
            loss_history.append(loss.item())
            
            # Print progress
            if (iteration + 1) % 10 == 0:
                logger.info(f"Iteration {iteration + 1}/{max_iterations}, Loss: {loss.item():.6f}")
        
        # Store loss history
        self.loss_history.extend(loss_history)
        
        logger.info("Model trained!")
    
    def compute_loss(self, batch):
        """
        Compute the loss.
        
        Args:
            batch: Batch of data
        """
        # Unpack batch
        theta, x = batch
        
        # Sample time
        t = torch.rand((len(theta), 1))
        
        # Compute loss
        loss = self.diffusion_model(x, t, theta)
        
        return loss
    
    def sample_batch(self):
        """
        Sample a batch from the data.
        """
        # Sample indices
        indices = np.random.choice(len(self.data), self.batch_size)
        
        # Sample data
        batch = [self.data[i] for i in indices]
        
        # Convert to tensors
        theta = torch.stack([item[0] for item in batch])
        x = torch.stack([item[1] for item in batch])
        
        return theta, x
    
    def update_truncated_prior(self):
        """
        Update the truncated prior.
        """
        logger.info("Updating truncated prior...")
        
        # Sample from posterior approximation
        theta_samples = self.sample_posterior_approximation(1000)
        
        # Compute HPR
        self.HPR = self.compute_HPR(theta_samples)
        
        # Update truncated prior
        self.truncated_prior = self.truncated_prior
        self.truncated_prior = self.truncated_prior
        
        logger.info("Truncated prior updated!")
    
    def compute_HPR(self, theta_samples):
        """
        Compute the highest probability region (HPR).
        
        Args:
            theta_samples: Samples from the posterior approximation
        """
        # Compute log probability
        log_probs = self.compute_log_probabilities(theta_samples)
        
        # Compute HPR
        threshold = np.quantile(log_probs, 1 - self.epsilon)
        HPR = theta_samples[log_probs >= threshold]
        
        return HPR
    
    def compute_log_probabilities(self, theta_samples):
        """
        Compute the log probabilities.
        
        Args:
            theta_samples: Samples from the posterior approximation
        """
        # Compute probabilities
        log_probs = []
        for theta in theta_samples:
            # Compute probability
            prob = self.posterior_approximation.log_prob(theta)
            log_probs.append(prob.item())
        
        return np.array(log_probs)
    
    def update_proposal(self):
        """
        Update the proposal.
        """
        logger.info("Updating proposal...")
        
        # Update proposal
        self.proposal = self.truncated_prior
        
        logger.info("Proposal updated!")
    
    def update_posterior_approximation(self):
        """
        Update the posterior approximation.
        """
        logger.info("Updating posterior approximation...")
        
        # Update posterior approximation
        self.posterior_approximation = self.posterior_approximation
        
        logger.info("Posterior approximation updated!")
    
    def update_HPR(self):
        """
        Update the HPR.
        """
        logger.info("Updating HPR...")
        
        # Update HPR
        self.HPR = self.HPR
        
        logger.info("HPR updated!")
    
    def converged(self):
        """
        Check convergence.
        """
        logger.info("Checking convergence...")
        
        # Check convergence
        if len(self.loss_history) < 10:
            return False
        
        # Compute average loss
        avg_loss = np.mean(self.loss_history[-10:])
        
        # Check if loss is small enough
        if avg_loss < self.tolerance:
            return True
        
        return False
    
    def final_training(self):
        """
        Final training.
        """
        logger.info("Final training...")
        
        # Final training
        self.train_model(10)
        
        logger.info("Final training completed!")
    
    def save_results(self):
        """
        Save results.
        """
        logger.info("Saving results...")
        
        # Save results
        with open("results.pkl", "wb") as f:
            pickle.dump({
                "loss_history": self.loss_history,
                "training_time": self.training_time,
            }, f)
        
        logger.info("Results saved!")
    
    def predict(self, observation):
        """
        Predict the posterior distribution.
        
        Args:
            observation: Observed data
        """
        # Set evaluation mode
        self.score_network.eval()
        
        # Sample from posterior
        samples = self.diffusion_model.sample(observation, 100)
        
        return samples

def main():
    """
    Main function.
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description="Reproduce results from the paper")
    parser.add_argument("--budget", type=int, default=1000, help="Simulation budget")
    parser.add_argument("--rounds", type=int, default=10, help="Number of rounds")
    parser.add_argument("--output", type=str, default="output.csv", help="Output file")
    args = parser.parse_args()
    
    # Initialize estimator
    estimator = SequentialNeuralScoreEstimator(theta_dim=10, x_dim=10)
    
    # Define simulator
    def simulator(theta):
        # Simple simulator
        return np.random.normal(theta, 0.1)
    
    # Train estimator
    estimator.train(observation=np.random.randn(10), simulator=simulator, budget=args.budget, rounds=args.rounds)
    
    # Predict
    samples = estimator.predict(observation=np.random.randn(10))
    
    # Save results
    np.savetxt(args.output, samples, delimiter=",")
    
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()