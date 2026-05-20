import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def generate_simulated_data(n_models=75, n_samples=1000):
    """
    Generate simulated data for 75 models with varying performance.
    This simulates the 75 models from the paper.
    """
    np.random.seed(42)
    
    # Generate ID accuracy (in-distribution accuracy)
    # VMs: 36 models with ID accuracy between 0.6 and 0.8
    # VLMs: 39 models with ID accuracy between 0.5 and 0.7
    vm_id_acc = np.random.uniform(0.6, 0.8, 36)
    vlm_id_acc = np.random.uniform(0.5, 0.7, 39)
    id_acc = np.concatenate([vm_id_acc, vlm_id_acc])
    
    # Generate LCA distance
    # Based on the paper, LCA distance is correlated with ID accuracy
    # For VMs: higher ID accuracy -> higher LCA distance
    # For VLMs: higher ID accuracy -> lower LCA distance
    lca_distance = np.zeros(n_models)
    for i in range(n_models):
        if i < 36:  # VMs
            # VMs: higher ID accuracy -> higher LCA distance
            lca_distance[i] = 10 - 10 * id_acc[i] + np.random.normal(0, 0.5)
        else:  # VLMs
            # VLMs: higher ID accuracy -> lower LCA distance
            lca_distance[i] = 5 * id_acc[i] + np.random.normal(0, 0.5)
    
    # Generate OOD accuracy
    # OOD accuracy is correlated with LCA distance
    # Higher LCA distance -> higher OOD accuracy
    ood_acc = np.zeros(n_models)
    for i in range(n_models):
        ood_acc[i] = 0.5 + 0.5 * (1 - lca_distance[i] / 10) + np.random.normal(0, 0.05)
    
    # Save data
    data = {
        'Model': [f'Model_{i}' for i in range(n_models)],
        'Model_Type': ['VM' if i < 36 else 'VLM' for i in range(n_models)],
        'ID_Accuracy': id_acc,
        'LCA_Distance': lca_distance,
        'OOD_Accuracy': ood_acc
    }
    
    return data

def main():
    parser = argparse.ArgumentParser(description='Generate data and plot results')
    parser.add_argument('--input', default='results/lca_results.csv', help='Input file')
    parser.add_argument('--output', default='results/plot.png', help='Output file')
    args = parser.parse_args()
    
    # Generate data
    data = generate_simulated_data()
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.input, index=False)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(df['LCA_Distance'], df['OOD_Accuracy'], alpha=0.7)
    
    # Fit a line
    z = np.polyfit(df['LCA_Distance'], df['OOD_Accuracy'], 1)
    p = np.poly1d(z)
    plt.plot(df['LCA_Distance'], p(df['LCA_Distance']), "r--")
    
    plt.xlabel('LCA Distance')
    plt.ylabel('OOD Accuracy')
    plt.title('LCA Distance vs. OOD Accuracy')
    plt.grid(True)
    plt.savefig(args.output)
    plt.show()
    
    print(f"Data saved to {args.input}")
    print(f"Plot saved to {args.output}")

if __name__ == "__main__":
    main()