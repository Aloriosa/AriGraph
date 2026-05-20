import os
import json
import csv
import numpy as np
import torch
from typing import Dict, List
import argparse

def load_results(results_dir: str) -> Dict:
    """Load evaluation results from CSV file"""
    csv_path = os.path.join(results_dir, 'evaluation_results.csv')
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Results file not found: {csv_path}")
    
    results = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                'task_id': int(row['task_id']),
                'average_accuracy': float(row['average_accuracy']),
                'final_accuracy': float(row['final_accuracy']),
                'forgetting': float(row['forgetting']),
                'total_parameters': int(row['total_parameters']),
                'expansion_count': int(row['expansion_count'])
            })
    
    return results

def calculate_metrics(results: List[Dict]) -> Dict:
    """Calculate comprehensive evaluation metrics"""
    if not results:
        return {}
    
    # Calculate average accuracy (A_n)
    avg_accuracy = np.mean([r['average_accuracy'] for r in results])
    
    # Calculate final accuracy (F_n)
    final_accuracy = results[-1]['final_accuracy']
    
    # Calculate average forgetting
    avg_forgetting = np.mean([r['forgetting'] for r in results])
    
    # Calculate total parameters
    total_parameters = results[-1]['total_parameters']
    
    # Calculate total expansions
    total_expansions = results[-1]['expansion_count']
    
    # Calculate forward transfer (average accuracy on tasks after their training)
    # For SEMA, we calculate the average accuracy on tasks 1 to n-1 after training task n
    forward_transfer = 0
    if len(results) > 1:
        # For each task after the first, calculate average accuracy on previous tasks
        for i in range(1, len(results)):
            # Average accuracy on tasks 0 to i-1 after training task i
            prev_accuracies = [results[j]['average_accuracy'] for j in range(i)]
            forward_transfer += np.mean(prev_accuracies)
        forward_transfer /= (len(results) - 1)
    
    # Calculate backward transfer (change in accuracy on previous tasks after new tasks)
    backward_transfer = 0
    if len(results) > 1:
        # Calculate average change in accuracy for each previous task
        changes = []
        for i in range(len(results) - 1):
            # Accuracy on task i after training task i (first time)
            initial_acc = results[i]['average_accuracy']
            # Accuracy on task i after all training
            final_acc = results[-1]['task_accuracy'][i] if 'task_accuracy' in results[-1] else results[i]['final_accuracy']
            changes.append(final_acc - initial_acc)
        backward_transfer = np.mean(changes) if changes else 0
    
    return {
        'average_accuracy': avg_accuracy,
        'final_accuracy': final_accuracy,
        'average_forgetting': avg_forgetting,
        'total_parameters': total_parameters,
        'total_expansions': total_expansions,
        'forward_transfer': forward_transfer,
        'backward_transfer': backward_transfer
    }

def main():
    parser = argparse.ArgumentParser(description='Evaluate SEMA results')
    parser.add_argument('--results_dir', type=str, default='results', help='Directory with results')
    parser.add_argument('--dataset', type=str, default='split_cifar', help='Dataset used')
    parser.add_argument('--num_tasks', type=int, default=10, help='Number of tasks')
    parser.add_argument('--output_file', type=str, default='evaluation_results.csv', help='Output file')
    
    args = parser.parse_args()
    
    # Load results
    results = load_results(args.results_dir)
    
    # Calculate metrics
    metrics = calculate_metrics(results)
    
    # Print summary
    print("=== SEMA EVALUATION RESULTS ===")
    print(f"Dataset: {args.dataset}")
    print(f"Number of tasks: {args.num_tasks}")
    print(f"Average Accuracy (A_n): {metrics['average_accuracy']:.4f}")
    print(f"Final Accuracy (F_n): {metrics['final_accuracy']:.4f}")
    print(f"Average Forgetting: {metrics['average_forgetting']:.4f}")
    print(f"Total Parameters: {metrics['total_parameters']:,}")
    print(f"Total Adapter Expansions: {metrics['total_expansions']}")
    print(f"Forward Transfer: {metrics['forward_transfer']:.4f}")
    print(f"Backward Transfer: {metrics['backward_transfer']:.4f}")
    print("==============================")
    
    # Save to file
    output_path = os.path.join(args.results_dir, args.output_file)
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        writer.writerow(['average_accuracy', metrics['average_accuracy']])
        writer.writerow(['final_accuracy', metrics['final_accuracy']])
        writer.writerow(['average_forgetting', metrics['average_forgetting']])
        writer.writerow(['total_parameters', metrics['total_parameters']])
        writer.writerow(['total_expansions', metrics['total_expansions']])
        writer.writerow(['forward_transfer', metrics['forward_transfer']])
        writer.writerow(['backward_transfer', metrics['backward_transfer']])
    
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()