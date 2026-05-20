import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
import os

def visualize_critical_states(critical_states_path, output="results/critical_states_visualization.png"):
    """
    Visualize critical states using t-SNE for dimensionality reduction.
    """
    # Load critical states
    with open(critical_states_path, 'rb') as f:
        critical_data = pickle.load(f)
    
    critical_states = critical_data['critical_states']
    critical_scores = critical_data['critical_scores']
    
    print(f"Visualizing {len(critical_states)} critical states with shape {critical_states.shape[1:]}")
    
    # If state dimension is too high, use t-SNE to reduce to 2D
    if critical_states.shape[1] > 2:
        from sklearn.manifold import TSNE
        
        # Sample a subset for visualization if too large
        max_samples = min(1000, len(critical_states))
        sample_indices = np.random.choice(len(critical_states), size=max_samples, replace=False)
        states_sample = critical_states[sample_indices]
        scores_sample = critical_scores[sample_indices]
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
        states_2d = tsne.fit_transform(states_sample)
        
        # Create scatter plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(states_2d[:, 0], states_2d[:, 1], c=scores_sample, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label='Importance Score')
        plt.title('t-SNE Visualization of Critical States')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.grid(True, alpha=0.3)
        
    else:
        # 2D states, direct plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(critical_states[:, 0], critical_states[:, 1], c=critical_scores, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label='Importance Score')
        plt.title('Critical States Visualization')
        plt.xlabel('State Dimension 1')
        plt.ylabel('State Dimension 2')
        plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.savefig(output, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Critical states visualization saved to {output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--critical_states_path', type=str, default='data/critical_states.pkl')
    parser.add_argument('--output', type=str, default='results/critical_states_visualization.png')
    
    args = parser.parse_args()
    
    visualize_critical_states(args.critical_states_path, args.output)