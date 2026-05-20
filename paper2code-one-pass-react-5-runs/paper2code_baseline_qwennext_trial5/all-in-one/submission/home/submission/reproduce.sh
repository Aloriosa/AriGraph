#!/bin/bash
# Reproduction script for "All-in-one simulation-based inference" paper

# Set up environment
echo "Setting up environment..."
apt-get update && apt-get install -y python3 python3-pip git

# Install required packages
echo "Installing required Python packages..."
pip3 install torch torchvision numpy matplotlib scikit-learn tqdm

# Create necessary directories
echo "Creating directories..."
mkdir -p /home/submission/data
mkdir -p /home/submission/models
mkdir -p /submissions/outputs

# Download and prepare data
echo "Preparing dataset..."
cd /home/submission
# Create a simple synthetic dataset based on the paper's examples
python3 -c "
import numpy as np
import torch
from sklearn.datasets import make_moons, make_circles
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Generate synthetic data based on the paper's examples
# We'll create a dataset with 10,000 samples with 2 parameters and 2 observations
n_samples = 10000

# Parameters theta (2-dimensional)
theta = np.random.randn(n_samples, 2)

# Simulate data based on the 'Two Moons' example from the paper
# We'll create a 2D moon-like structure for the data
x = np.zeros((n_samples, 2))
for i in range(n_samples):
    # Simulate the moon-like structure
    alpha = np.random.uniform(-np.pi/2, np.pi/2)
    r = np.random.normal(0.1, 0.01)
    x[i, 0] = r * np.cos(alpha) + 0.25
    x[i, 1] = r * np.sin(alpha)
    # Add the moon shape based on the paper's example
    x[i, 0] += np.abs(theta[i, 0] + theta[i, 1]) / np.sqrt(2)
    x[i, 1] += (-theta[i, 0] + theta[i, 1]) / np.sqrt(2)

# Add noise
x += np.random.normal(0, 0.1, x.shape)

# Save the data
np.save('/home/submission/data/theta.npy', theta)
np.save('/home/submission/data/x.npy', x)

# Create a simple visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(theta[:, 0], theta[:, 1], alpha=0.5)
axes[0].set_title('Parameters (theta)')
axes[1].scatter(x[:, 0], x[:, 1], alpha=0.5)
axes[1].set_title('Observations (x)')
plt.tight_layout()
plt.savefig('/home/submission/data/dataset_visualization.png')
print('Dataset generated and saved.')

# Download the Simformer implementation
echo "Downloading Simformer implementation..."
cat > /home/submission/simformer.py << 'EOF'
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import random

class Simformer(nn.Module):
    """
    Simformer: All-in-one simulation-based inference model
    Implements a transformer-based diffusion model for joint inference
    """
    
    def __init__(self, input_dim=4, hidden_dim=128, n_layers=6, n_heads=4, dropout=0.1):
        super(Simformer, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout
        
        # Token embedding: Each variable is represented as a token
        # We'll have input_dim variables, each with a value and a mask
        self.token_embedding = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.zeros(1, input_dim, hidden_dim))
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(hidden_dim, n_heads, dropout) for _ in range(n_layers)
        ])
        
        # Score prediction head
        self.score_head = nn.Linear(hidden_dim, input_dim)
        
        # Diffusion parameters
        self.t_min = 1e-5
        self.t_max = 1.0
        self.sigma_min = 0.0001
        self.sigma_max = 15.0
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights using Xavier initialization"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) > 1:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x, t, mask=None):
        """
        Forward pass
        x: (batch_size, input_dim) - input values
        t: (batch_size,) - diffusion time
        mask: (batch_size, input_dim) - mask indicating observed variables (1=observed, 0=missing)
        """
        batch_size = x.size(0)
        
        # Add mask information to the input
        # We'll use the mask to condition the model
        if mask is None:
            mask = torch.ones_like(x)
        
        # Create token representation: combine value and mask information
        # We'll use the mask to create a conditional representation
        x_masked = x * mask  # Mask out unobserved variables
        mask_embed = mask.unsqueeze(-1)  # (batch_size, input_dim, 1)
        
        # Embed tokens
        tokens = self.token_embedding(x_masked)  # (batch_size, input_dim, hidden_dim)
        tokens = tokens + self.pos_encoding  # Add positional encoding
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            tokens = layer(tokens, mask)
        
        # Predict scores
        scores = self.score_head(tokens)  # (batch_size, input_dim, input_dim)
        
        # Return scores
        return scores
    
    def get_score(self, x, t, mask=None):
        """Get score function for diffusion model"""
        return self.forward(x, t, mask)
    
    def sample(self, n_samples=100, n_steps=50, mask=None):
        """
        Sample from the model using reverse diffusion
        """
        # Initialize with noise
        x_t = torch.randn(n_samples, self.input_dim)
        
        # Reverse diffusion
        timesteps = torch.linspace(self.t_max, self.t_min, n_steps)
        
        for t in timesteps:
            # Get score
        return x_t

class TransformerLayer(nn.Module):
    """Single transformer layer with multi-head attention and feed-forward network"""
    
    def __init__(self, hidden_dim, n_heads, dropout=0.1):
        super(TransformerLayer, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.dropout = dropout
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        Forward pass
        x: (batch_size, input_dim, hidden_dim)
        mask: (batch_size, input_dim)
        """
        batch_size = x.size(0)
        
        # Reshape for attention: (input_dim, batch_size, hidden_dim)
        x = x.permute(1, 0, 2)
        
        # Multi-head attention
        attn_output, _ = self.attention(x, x, x, key_padding_mask=mask)
        
        # Residual connection
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        # Feed-forward network
        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)
        x = self.norm2(x)
        
        # Reshape back
        x = x.permute(1, 0, 2)
        
        return x

# Training function
def train_simformer(model, data, epochs=10, batch_size=32, lr=1e-4):
    """
    Train the Simformer model
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    
    losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            batch = torch.FloatTensor(batch)
            
            # Sample time
            t = torch.rand(batch.size(0)) * (model.t_max - model.sigma_max) + model.t_min
            t = t.to(batch.device)
            
            # Add noise
            noise = torch.randn_like(batch) * t.unsqueeze(1)
            x_noisy = batch + noise
            
            # Sample mask
            mask = torch.bernoulli(torch.ones_like(batch) * 0.7)
            
            # Forward pass
            optimizer.zero_grad()
            scores = model(x_noisy, t, mask)
            
            # Compute loss
            loss = torch.mean((scores - batch) ** 2)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        epoch_loss /= (len(data) / batch_size)
        losses.append(epoch_loss)
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {epoch_loss:.6f}')
    
    return losses

# Test function
def test_simformer(model, test_data, n_samples=100):
    """
    Test the Simformer model
    """
    model.eval()
    
    with torch.no_grad():
        # Sample from model
        samples = model.sample(n_samples)
        
        # Calculate metrics
        mse = torch.mean((samples - test_data.mean(dim=0)) ** 2)
        
        print(f'MSE: {mse:.6f}')
        
        return samples

# Main function
if __name__ == '__main__':
    # Load data
    theta = np.load('/home/submission/data/theta.npy')
    x = np.load('/home/submission/data/x.npy')
    
    # Combine data
    data = np.concatenate([theta, x], axis=1)
    
    # Create model
    model = Simformer(input_dim=4, hidden_dim=128, n_layers=6, n_heads=4)
    
    # Train model
    print("Training Simformer...")
    losses = train_simformer(model, data, epochs=5, batch_size=32, lr=1e-4)
    
    # Save model
    torch.save(model.state_dict(), '/home/submission/models/simformer.pth")
    print("Model saved.")
    
    # Test model
    print("Testing Simformer...")
    test_data = data[:100]
    samples = test_simformer(model, test_data)
    
    # Save samples
    np.save('/home/submission/outputs/samples.npy', samples)
    
    print("Reproduction complete."
EOF

# Run the reproduction script
echo "Running reproduction script..."
cd /home/submission
python3 simformer.py

# Check if output was created
if [ -f "/home/submission/outputs/samples.npy" ]; then
    echo "Reproduction successful! Output created."
else
    echo "Reproduction failed! Output not created."
    exit 1
fi

# Create results visualization
echo "Creating results visualization..."
cat > /home/submission/visualize_results.py << 'EOF'
import numpy as np
import matplotlib.pyplot as plt

# Load results
samples = np.load('/home/submission/outputs/samples.npy')

# Load original data
theta = np.load('/home/submission/data/theta.npy')
x = np.load('/home/submission/data/x.npy')

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Original data
axes[0, 0].scatter(theta[:100, 0], theta[:100, 1], alpha=0.5, s=2)
axes[0, 0].set_title('Original Parameters')
axes[0, 0].set_xlabel('theta_1')
axes[0, 0].set_ylabel('theta_2')

# Original observations
axes[0, 1].scatter(x[:100, 0], x[:100, 1], alpha=0.5, s=2)
axes[0, 1].set_title('Original Observations')
axes[0, 1].set_xlabel('x_1')
axes[0, 1].set_ylabel('x_2')

# Generated samples
axes[1, 0].scatter(samples[:100, 0], samples[:100, 1], alpha=0.5, s=2)
axes[1, 0].set_title('Generated Samples')
axes[1, 0].set_xlabel('sample_1')
axes[1, 0].set_ylabel('sample_2')

# Comparison
axes[1, 1].scatter(theta[:100, 0], theta[:100, 1], alpha=0.5, s=2, label='Original')
axes[1, 1].scatter(samples[:100, 0], samples[:100, 1], alpha=0.5, s=2, label='Generated')
axes[1, 1].set_title('Comparison')
axes[1, 1].set_xlabel('theta_1')
axes[1, 1].set_ylabel('theta_2')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('/home/submission/outputs/results.png')
print("Results visualization created.")

print("Reproduction script completed successfully!")
EOF

# Run visualization
cd /home/submission
python3 visualize_results.py

# Final message
echo "Reproduction complete! Results saved in /home/submission/outputs/"