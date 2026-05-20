import numpy as np
import pandas as pd
import pickle
import argparse
import os

def load_hierarchy(hierarchy_file):
    """Load the class hierarchy from a pickle file."""
    with open(hierarchy_file, 'rb') as f:
        hierarchy = pickle.load(f)
    return hierarchy

def find_lca(hierarchy, class1, class2):
    """Find the Lowest Common Ancestor (LCA) of two classes in the hierarchy."""
    # This is a simplified version. In a real implementation, we would use a tree structure.
    # For this reproduction, we assume the hierarchy is a dictionary of parent-child relationships.
    def get_path_to_root(class_name):
        path = []
        current = class_name
        while current is not None:
            path.append(current)
            current = hierarchy.get(current, None)
        return path
    
    path1 = get_path_to_root(class1)
    path2 = get_path_to_root(class2)
    
    # Find the last common node in the paths
    lca = None
    for node1, node2 in zip(path1, path2):
        if node1 == node2:
            lca = node1
        else:
            break
    return lca

def calculate_lca_distance(predictions, labels, hierarchy):
    """Calculate the LCA distance for a set of predictions and labels."""
    n = len(predictions)
    total_distance = 0
    correct = 0
    
    for i in range(n):
        pred = predictions[i]
        true = labels[i]
        
        if pred == true:
            # Correct prediction
            correct += 1
        else:
            # Incorrect prediction
            lca = find_lca(hierarchy, pred, true)
            # Calculate the distance based on the hierarchy
            # In a real implementation, this would be the depth of the LCA
            distance = 1  # Simplified: use the depth difference
            total_distance += distance
    
    return total_distance / n if n > 0 else 0

def main():
    parser = argparse.ArgumentParser(description='Calculate LCA distance for model predictions')
    parser.add_argument('--input_data', required=True, help='Path to the input predictions file')
    parser.add_argument('--labels', required=True, help='Path to the labels file')
    parser.add_argument('--hierarchy', required=True, help='Path to the hierarchy file')
    parser.add_argument('--output', default='lca_results.csv', help='Output file path')
    args = parser.parse_args()
    
    # Load data
    predictions = np.load(args.input_data)
    labels = np.load(args.labels)
    hierarchy = load_hierarchy(args.hierarchy)
    
    # Calculate LCA distance
    lca_distance = calculate_lca_distance(predictions, labels, hierarchy)
    
    # Calculate accuracy
    accuracy = np.mean(predictions == labels)
    
    # Save results
    results = pd.DataFrame({
        'LCA_Distance': [lca_distance],
        'Top1_Accuracy': [accuracy]
    })
    
    results.to_csv(args.output, index=False)
    print(f"Results saved to {args.output}")
    print(f"LCA Distance: {lca_distance:.4f}")
    print(f"Top-1 Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()