import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import argparse
import os
from reward_encoder import TransformerRewardEncoder

def visualize_latent_space(
    reward_encoder_path: str,
    data_path: str,
    output: str = 'results/latent_space_tsne.png'
):
    """
    Visualize the latent space of the reward encoder using t-SNE.
    
    Args:
        reward_encoder_path: Path to trained reward encoder
        data_path: Path to offline trajectories data
        output: Path to save visualization
    """
    # Load data
    with open(data_path, 'rb') as f:
        trajectories = pickle.load(f)
    
    # Get state dimension
    state_dim = trajectories[0]['states'].shape[1]
    latent_dim = 128  # From training configuration
    
    # Load reward encoder
    reward_encoder = TransformerRewardEncoder(
        state_dim=state_dim,
        latent_dim=latent_dim
    )
    reward_encoder.load_state_dict(torch.load(reward_encoder_path, map_location='cpu'))
    reward_encoder.eval()
    
    # Extract latent vectors and task types
    latent_vectors = []
    task_types = []
    
    for traj in trajectories:
        states = traj['states']
        rewards = traj['rewards']
        reward_type = traj['reward_function']['type']
        
        # Use first 10 state-reward pairs to encode the reward function
        # In practice, we'd use the entire trajectory
        seq_len = min(len(states), 10)
        states_subset = states[:seq_len]
        rewards_subset = rewards[:seq_len]
        
        states_tensor = torch.from_numpy(states_subset).float()
        rewards_tensor = torch.from_numpy(rewards_subset).float()
        
        with torch.no_grad():
            z = reward_encoder.encode(states_tensor.unsqueeze(0), rewards_tensor.unsqueeze(0))
            z = z.squeeze().cpu().numpy()
        
        latent_vectors.append(z)
        task_types.append(reward_type)
    
    # Convert to numpy array
    latent_vectors = np.array(latent_vectors)
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    latent_2d = tsne.fit_transform(latent_vectors)
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Get unique task types
    unique_types = list(set(task_types))
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_types)))
    
    # Plot each task type with different color
    for i, task_type in enumerate(unique_types):
        mask = [t == task_type for t in task_types]
        plt.scatter(latent_2d[mask, 0], latent_2d[mask, 1], 
                   c=[colors[i]], label=task_type, alpha=0.7, s=50)
    
    plt.title('t-SNE Visualization of Latent Reward Encodings')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    os.makedirs(os.path.dirname(output), exist_ok=True)
    plt.savefig(output, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Latent space visualization saved to {output}")
    
    # Print some statistics
    print(f"Total latent vectors: {len(latent_vectors)}")
    print(f"Unique task types: {len(unique_types)}")
    for task_type in unique_types:
        count = task_types.count(task_type)
        print(f"  {task_type}: {count} samples")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize latent space')
    parser.add_argument('--reward_encoder_path', type=str, default='models/reward_encoder.pth',
                        help='Path to trained reward encoder')
    parser.add_argument('--data_path', type=str, default='data/offline_trajectories.pkl',
                        help='Path to offline trajectories data')
    parser.add_argument('--output', type=str, default='results/latent_space_tsne.png',
                        help='Path to save visualization')
    
    args = parser.parse_args()
    
    visualize_latent_space(
        reward_encoder_path=args.reward_encoder_path,
        data_path=args.data_path,
        output=args.output
    )