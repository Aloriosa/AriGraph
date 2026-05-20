import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np
import argparse
import os
from reward_encoder import TransformerRewardEncoder
from torch.utils.data import DataLoader, Dataset

class TrajectoryDataset(Dataset):
    """Dataset for loading offline trajectories."""
    
    def __init__(self, data_path: str, max_seq_len: int = 100):
        with open(data_path, 'rb') as f:
            self.trajectories = pickle.load(f)
        
        self.max_seq_len = max_seq_len
        self.samples = self._prepare_samples()
    
    def _prepare_samples(self) -> List[Dict]:
        """Prepare state-reward pairs from trajectories."""
        samples = []
        
        for traj in self.trajectories:
            states = traj['states']  # Shape: (seq_len, state_dim)
            rewards = traj['rewards']  # Shape: (seq_len,)
            
            # Limit sequence length
            seq_len = min(len(states), self.max_seq_len)
            states = states[:seq_len]
            rewards = rewards[:seq_len]
            
            # Create sample
            samples.append({
                'states': states,
                'rewards': rewards,
                'seq_len': seq_len
            })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        sample = self.samples[idx]
        states = torch.from_numpy(sample['states']).float()
        rewards = torch.from_numpy(sample['rewards']).float()
        seq_len = sample['seq_len']
        
        return states, rewards, seq_len

def collate_fn(batch):
    """Custom collate function to handle variable-length sequences."""
    states_list, rewards_list, seq_lens = zip(*batch)
    
    # Find max sequence length
    max_len = max(seq_lens)
    
    # Pad sequences
    padded_states = []
    padded_rewards = []
    
    for states, rewards in zip(states_list, rewards_list):
        seq_len = states.shape[0]
        padding = max_len - seq_len
        
        # Pad states
        if padding > 0:
            padded_state = F.pad(states, (0, 0, 0, padding))
        else:
            padded_state = states
        padded_states.append(padded_state)
        
        # Pad rewards
        if padding > 0:
            padded_reward = F.pad(rewards, (0, padding))
        else:
            padded_reward = rewards
        padded_rewards.append(padded_reward)
    
    # Stack tensors
    states_batch = torch.stack(padded_states)
    rewards_batch = torch.stack(padded_rewards)
    
    return states_batch, rewards_batch

def train_reward_encoder(
    data_path: str,
    model_path: str,
    latent_dim: int = 128,
    num_layers: int = 4,
    num_heads: int = 8,
    hidden_dim: int = 256,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    epochs: int = 50,
    device: str = 'cuda'
):
    """
    Train the transformer-based reward encoder VAE.
    
    Args:
        data_path: Path to offline trajectories data
        model_path: Path to save trained model
        latent_dim: Dimension of latent space
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        hidden_dim: Hidden dimension in transformer
        batch_size: Batch size
        learning_rate: Learning rate
        epochs: Number of training epochs
        device: Device to train on ('cuda' or 'cpu')
    """
    # Set device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    dataset = TrajectoryDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    # Get state dimension from first sample
    state_dim = dataset[0][0].shape[1]
    
    # Initialize model
    model = TransformerRewardEncoder(
        state_dim=state_dim,
        latent_dim=latent_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim
    ).to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        
        for states, rewards in dataloader:
            states = states.to(device)
            rewards = rewards.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            mu, logvar, reconstructed_rewards = model(states, rewards)
            
            # Reconstruction loss (MSE)
            recon_loss = F.mse_loss(reconstructed_rewards, rewards, reduction='mean')
            
            # KL divergence loss
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / states.size(0)
            
            # Total loss
            loss = recon_loss + kl_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
        
        avg_loss = total_loss / len(dataloader)
        avg_recon_loss = total_recon_loss / len(dataloader)
        avg_kl_loss = total_kl_loss / len(dataloader)
        
        print(f"Epoch [{epoch+1}/{epochs}], "
              f"Loss: {avg_loss:.4f}, "
              f"Recon Loss: {avg_recon_loss:.4f}, "
              f"KL Loss: {avg_kl_loss:.4f}")
    
    # Save model
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train reward encoder')
    parser.add_argument('--data_path', type=str, default='data/offline_trajectories.pkl',
                        help='Path to offline trajectories data')
    parser.add_argument('--model_path', type=str, default='models/reward_encoder.pth',
                        help='Path to save trained model')
    parser.add_argument('--latent_dim', type=int, default=128,
                        help='Dimension of latent space')
    parser.add_argument('--num_layers', type=int, default=4,
                        help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Hidden dimension in transformer')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to train on')
    
    args = parser.parse_args()
    
    # Create model directory if it doesn't exist
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    
    train_reward_encoder(
        data_path=args.data_path,
        model_path=args.model_path,
        latent_dim=args.latent_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        device=args.device
    )