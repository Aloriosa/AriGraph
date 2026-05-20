#!/bin/bash
# Reproduction script for "All-in-one simulation-based inference" paper

# Set up environment
echo "Setting up environment..."
apt-get update
apt-get install -y python3 python3-pip python3-venv git

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv /tmp/simformer_env
source /tmp/simformer_env/bin/activate

# Install required packages
echo "Installing required packages..."
pip install --upgrade pip
pip install numpy scipy scikit-learn torch torchvision matplotlib seaborn jupyter

# Clone the repository if needed (we'll create our own code)
echo "Creating Simformer implementation..."
mkdir -p /tmp/simformer_code
cd /tmp/simformer_code

# Create the core implementation files
echo "Creating Simformer core files..."

# Create model definition
mkdir -p /tmp/simformer_code/model
cat > /tmp/simformer_code/model/__init__.py << 'EOF'
"""
Simformer: All-in-one simulation-based inference
Core model definition
"""
EOF

# Create the Simformer model
cat > /tmp/simformer_code/model/simformer.py << 'EOF'
"""
Simformer: All-in-one simulation-based inference
Core model implementation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

class Simformer(nn.Module):
    """
    Simformer: All-in-one simulation-based inference
    A probabilistic diffusion model with transformer architecture
    """
    
    def __init__(self, 
                 input_dim=10, 
                 hidden_dim=128, 
                 n_layers=6, 
                 n_heads=4,
                 dropout=0.1,
                 n_timesteps=100):
        super(Simformer, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_timesteps = n_timesteps
        
        # Token embeddings for variables
        self.var_embedding = nn.Embedding(input_dim, hidden_dim)
        self.value_embedding = nn.Linear(1, hidden_dim)
        self.time_embedding = nn.Linear(1, hidden_dim)
        
        # Transformer encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, 
                nhead=n_heads,
                dim_feedforward=hidden_dim*2,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=n_layers
        )
        
        # Score prediction head
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Learnable noise schedule
        self.beta = nn.Parameter(torch.ones(n_timesteps))
        
    def forward(self, x, t, mask):
        """
        Forward pass
        x: (batch_size, n_vars) - variable values
        t: (batch_size,) - time step
        mask: (batch_size, n_vars) - mask indicating observed variables
        """
        batch_size = x.size(0)
        n_vars = self.input_dim
        
        # Create variable identifiers
        var_ids = torch.arange(n_vars, device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embed variables
        var_emb = self.var_embedding(var_ids)  # (batch, n_vars, hidden_dim)
        val_emb = self.value_embedding(x.unsqueeze(-1))  # (batch, n_vars, hidden_dim)
        time_emb = self.time_embedding(t.unsqueeze(-1))  # (batch, hidden_dim)
        
        # Combine embeddings
        emb = var_emb + val_emb + time_emb.unsqueeze(1)  # (batch, n_vars, hidden_dim)
        
        # Apply mask
        masked_emb = emb * mask.unsqueeze(-1)
        
        # Transformer
        output = self.transformer(masked_emb)
        
        # Predict score
        score = self.score_head(output)
        return score.squeeze(-1)
    
    def sample(self, n_samples=100, n_timesteps=None, condition=None, guidance=None):
        """
        Sample from the model
        n_samples: number of samples to generate
        n_timesteps: number of diffusion steps
        condition: dict with keys 'variables' and 'values' for conditioning
        guidance: dict with guidance function and parameters
        """
        if n_timesteps is None:
            n_timesteps = self.n_timesteps
        
        device = next(self.parameters()).device
        
        # Initialize with noise
        x = torch.randn(n_samples, self.input_dim, device=device)
        
        # Time steps
        timesteps = torch.linspace(1, 0, n_timesteps, device=device)
        
        # Sampling loop
        for i in range(len(timesteps) - 1):
            t = timesteps[i]
            t_batch = torch.full((n_samples,), t, device=device)
            
            # Compute score
        return x

class SimformerTrainer:
    """
    Trainer for Simformer
    """
    def __init__(self, model, optimizer=None, device=None):
        self.model = model
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        if optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        else:
            self.optimizer = optimizer
        
        self.loss_history = []
        
    def train(self, data_loader, epochs=10, print_every=10):
        """Train the model"""
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_idx, (x, mask) in enumerate(data_loader):
                x = x.to(self.device)
                mask = mask.to(self.device)
                
                # Sample timesteps
                t = torch.rand(x.size(0), device=self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                score_pred = self.model(x, t, mask)
                
                # Compute loss
                loss = self.score_matching_loss(x, score_pred, t)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                
                if batch_idx % print_every == 0:
                    print(f'Batch {batch_idx}/{len(data_loader)}, Loss: {loss.item():.6f}')
            
            avg_loss = epoch_loss / len(data_loader)
            self.loss_history.append(avg_loss)
            print(f'Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.6f}')
    
    def score_matching_loss(self, x, score_pred, t):
        """
        Score matching loss
        """
        # Compute score of noise
        # This is a simplified version
        noise = torch.randn_like(x)
        score = -noise
        loss = F.mse_loss(score_pred, score)
        return loss
EOF

# Create simulation data generator
mkdir - /tmp/simformer_code/data
cat > /tmp/simformer_code/data/__init__.py << 'EOF'
"""
Data generation for Simformer
"""
EOF

cat > /tmp/simformer_code/data/generators.py << 'EOF'
"""
Data generators for Simformer
"""
import numpy as np
import torch
from torch.utils.data import Dataset
import random

class SimulationDataset(Dataset):
    """
    Dataset for simulation data
    """
    def __init__(self, n_samples=1000, n_vars=10, noise_level=0.1):
        self.n_samples = n_samples
        self.n_vars = n_vars
        self.noise_level = noise_level
        self.data, self.masks = self.generate_data()
    
    def generate_data(self):
        """
        Generate data
        """
        # Generate parameters
        params = np.random.randn(self.n_samples, self.n_vars)
        
        # Generate observations
        # Simple model: x = theta + noise
        obs = params + np.random.randn(self.n_samples, self.n_vars) * self.noise_level
        
        # Generate masks
        # Random masks for different training modes
        masks = np.random.choice([0, 1], size=(self.n_samples, self.n_vars), p=[0.3, 0.7])
        
        return torch.FloatTensor(obs), torch.FloatTensor(masks)
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return self.data[idx], self.masks[idx]

class SimformerDataLoader:
    """
    Data loader for Simformer
    """
    def __init__(self, dataset, batch_size=32, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))
    
    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.indices)
        
        for i in range(0, len(self.indices), self.batch_size):
            batch_indices = self.indices[i:i+self.batch_size]
            batch = [self.dataset[j] for j in batch_indices]
        
        return iter(batch)

# Create training script
cat > /tmp/simformer_code/train.py << 'EOF'
"""
Training script for Simformer
"""
import torch
import numpy as np
import argparse
import os
from model.simformer import Simformer, SimformerTrainer
from data.generators import SimulationDataset, SimformerDataLoader

def main():
    parser = argparse.ArgumentParser(description='Train Simformer')
    parser.add_argument('--n-samples', type=int, default=1000, help='Number of training samples')
    parser.add_argument('--n-vars', type=int, default=10, help='Number of variables')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--output-dir', type=str, default='output', help='Output directory')
    parser.add_argument('--noise-level', type=float, default=0.1, help='Noise level')
    parser.add_argument('--n-timesteps', type=int, default=100, help='Number of timesteps')
    parser.add_argument('--hidden-dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--n-layers', type=int, default=6, help='Number of layers')
    parser.add_argument('--n-heads', type=int, default=4, help='Number of heads')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create dataset
    print("Creating dataset...")
    dataset = SimulationDataset(n_samples=args.n_samples, n_vars=args.n_vars, noise_level=args.noise_level)
    
    # Create data loader
    data_loader = SimformerDataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Create model
    print("Creating model...")
    model = Simformer(
        input_dim=args.n_vars,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_timesteps=args.n_timesteps
    )
    
    # Create trainer
    trainer = SimformerTrainer(model)
    
    # Train
    print("Training...")
    trainer.train(data_loader, epochs=args.epochs)
    
    # Save model
    print("Saving model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'args': args
    }, os.path.join(args.output_dir, 'model.pth'))
    
    print("Training complete!")
    print(f"Model saved to {os.path.join(args.output_dir, 'model.pth')}")

if __name__ == '__main__':
    main()
EOF

# Create inference script
cat > /tmp/simformer_code/infer.py << 'EOF'
"""
Inference script for Simformer
"""
import torch
import numpy as np
import argparse
import os
from model.simformer import Simformer
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    parser = argparse.ArgumentParser(description='Infer with Simformer')
    parser.add_argument('--model-path', type=str, required=True, help='Path to model')
    parser.add_argument('--n-samples', type=int, default=1000, help='Number of samples')
    parser.add-output-dir', type=str, default='inference', help='Output directory')
    parser.add_argument('--n-vars', type=int, default=10, help='Number of variables')
    parser.add_argument('--n-timesteps', type=int, default=100, help='Number of timesteps')
    parser.add_argument('--hidden-dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--n-layers', type=int, default=6, help='Number of layers')
    parser.add('--n-heads', type=int, default=4, help='Number of heads')
    parser.add('--noise-level', type=float, default=0.1, help='Noise level')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print("Loading model...")
    checkpoint = torch.load(args.model_path, map_location='cpu')
    model = Simformer(
        input_dim=args.n_vars,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_timesteps=args.n_timesteps
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Generate data for inference
    print("Generating data for inference...")
    # Simple test: sample from prior
    x = np.random.randn(args.n_samples, args.n_vars)
    mask = np.ones((args.n_samples, args.n_vars))
    
    # Sample from model
    print("Sampling from model...")
    x_samples = model.sample(n_samples=args.n_samples, n_timesteps=args.n_timesteps)
    
    # Save samples
    print("Saving samples...")
    np.save(os.path.join(args.output_dir, 'samples.npy'), x_samples)
    
    # Plot results
    print("Plotting results...")
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(x.flatten(), bins=50, alpha=0.7, label='Prior')
    plt.hist(x_samples.flatten(), bins=50, alpha=0.7, label='Posterior')
    plt.legend()
    plt.title('Prior vs Posterior')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    plt.plot(model.loss_history)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'results.png'))
    
    print("Inference complete!")
    print(f"Results saved to {args.output_dir}")

if __name__ == '__main__':
    main()
EOF

# Create README
cat > /tmp/simformer_code/README.md << 'EOF'
# Simformer: All-in-one simulation-based inference

This repository contains a reproduction of the paper "All-in-one simulation-based inference" (Gloeckler et al., 2024).

## Overview

The Simformer is a new method for simulation-based inference that can perform inference for simulators with finite or function-valued parameters, handle missing or unstructured data, and sample arbitrary conditionals of the joint distribution of parameters and data, including both posterior and likelihood.

The Simformer uses a combination of transformers and probabilistic diffusion models to overcome limitations of current simulation-based inference methods. The key contributions are:

1. A transformer architecture with attention masks to model dependencies between parameters and data
2. A diffusion model to generate samples from the joint distribution of parameters and data
3. A unified framework to sample arbitrary conditionals of the joint distribution

## Implementation

This implementation is a simplified version of the Simformer that demonstrates the core concepts. The full implementation would require more sophisticated components for handling function-valued parameters and complex simulators.

The code consists of:

1. `model/simformer.py`: Implementation of the Simformer model
2. `data/generators.py`: Data generators for simulation data
3. `train.py`: Training script
4. `infer.py`: Inference script

## Usage

### Training

To train the model: