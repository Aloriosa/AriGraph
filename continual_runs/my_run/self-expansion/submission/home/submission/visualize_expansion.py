#!/usr/bin/env python3
"""
Visualize adapter expansion patterns
"""
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from sema_model import SEMA

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize adapter expansion patterns')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained SEMA model')
    parser.add_argument('--output_file', type=str, default='results/expansion_visualization.png',
                       help='Output file for visualization')
    
    return parser.parse_args()

def load_expansion_history(model_path):
    """
    Load adapter expansion history from model checkpoint
    """
    checkpoint = torch.load(model_path, map_location='cpu')
    
    if isinstance(checkpoint, dict) and 'adapter_expansion_history' in checkpoint:
        return checkpoint['adapter_expansion_history']
    elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # Check if expansion history is in the model state dict
        return checkpoint.get('adapter_expansion_history', {})
    else:
        # Try to load from a separate JSON file
        json_path = model_path.replace('.pt', '_expansion_history.json')
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                return json.load(f)
        else:
            return {}

def visualize_expansion(expansion_history, output_file):
    """
    Create visualization of adapter expansion patterns
    """
    if not expansion_history:
        print("No expansion history found")
        return
    
    # Create a matrix showing when adapters were added
    num_layers = max(expansion_history.keys()) + 1 if expansion_history else 0
    max_adapters = max(len(adapters) for adapters in expansion_history.values()) if expansion_history else 0
    
    expansion_matrix = np.zeros((num_layers, max_adapters))
    
    for layer_idx, task_ids in expansion_history.items():
        for adapter_idx, task_id in enumerate(task_ids):
            if adapter_idx < max_adapters:
                expansion_matrix[layer_idx, adapter_idx] = task_id + 1  # +1 for 1-based indexing
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create heatmap
    im = ax.imshow(expansion_matrix, cmap='viridis', aspect='auto', origin='lower')
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Task ID', rotation=-90, va="bottom")
    
    # Set ticks
    ax.set_xlabel('Adapter Index')
    ax.set_ylabel('Transformer Layer')
    ax.set_title('Adapter Expansion Pattern Over Tasks')
    
    # Set tick labels
    ax.set_xticks(range(max_adapters))
    ax.set_xticklabels([f'Adapter {i}' for i in range(max_adapters)])
    
    ax.set_yticks(range(num_layers))
    ax.set_yticklabels([f'Layer {i}' for i in range(num_layers)])
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add value annotations
    for i in range(num_layers):
        for j in range(max_adapters):
            if expansion_matrix[i, j] > 0:
                ax.text(j, i, f'{int(expansion_matrix[i, j])}', 
                       ha="center", va="center", color="white", fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Expansion visualization saved to {output_file}")

def main():
    args = parse_args()
    
    # Load expansion history
    expansion_history = load_expansion_history(args.model_path)
    
    # Create visualization
    visualize_expansion(expansion_history, args.output_file)
    
    # Print summary
    if expansion_history:
        total_adapters = sum(len(adapters) for adapters in expansion_history.values())
        print(f"Total adapters added: {total_adapters}")
        print(f"Layers with expansion: {len(expansion_history)} out of {max(expansion_history.keys()) + 1 if expansion_history else 0}")
        
        for layer_idx, task_ids in expansion_history.items():
            print(f"Layer {layer_idx}: {len(task_ids)} adapters added at tasks {task_ids}")
    else:
        print("No expansion history found")

if __name__ == "__main__":
    main()