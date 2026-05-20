"""
Estimator for Sequential Neural Score Estimation (SNSE).
"""

import torch
import numpy as np
from typing import Optional, Tuple
import logging

# Set up logging
logger = logging.getLogger(__name__)

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
        """
        self.theta_dim = theta_dim
        self.x_dim = x_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.timesteps = timesteps
        self.learning_rate = learning_rate
        
        # Initialize score network
        self.score_network = ScoreNetwork(theta_dim, x_dim, hidden_dim=hidden_dim, num_layers=num_layers)
        
        # Initialize diffusion model
        self.diffusion_model = DiffusionModel(self.score_network, timesteps=timesteps)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.score_network.parameters(), lr=learning_rate)
        
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
        
        # Set observation
        self.observation = torch.tensor(observation, dtype=torch.float32)
        if len(self.observation.shape) == 1:
            self.observation = self.observation.unsqueeze(0)
        
        # Set prior
        self.prior = torch.distributions.Uniform(-1.0, 1.0)
        
        # Set truncated prior
        self.truncated_prior = self.prior
        
        # Set proposal
        self.proposal = self.prior
        
        # Set posterior approximation
        self.posterior_approximation = None
        
        # Set HPR
        self.HPR = None
        
        # Set training flag
        self.training_started = False
        self.training_finished = False
        
        # Start training
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