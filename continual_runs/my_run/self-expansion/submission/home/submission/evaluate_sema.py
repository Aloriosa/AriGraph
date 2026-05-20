#!/usr/bin/env python3
"""
Evaluate SEMA model on continual learning tasks
"""
import os
import argparse
import torch
import pickle
import numpy as np
import json
from sema_model import SEMA
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate SEMA model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained SEMA model')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['split_cifar100', 'split_tiny_imagenet'],
                       help='Dataset to evaluate on')
    parser.add_argument('--num_tasks', type=int, required=True,
                       help='Number of tasks in the dataset')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--output_file', type=str, default='results.csv',
                       help='Output file for results')
    
    return parser.parse_args()

def load_dataset(dataset_name, data_dir="./data"):
    """
    Load dataset for evaluation
    """
    dataset_path = f"{data_dir}/{dataset_name}_tasks.pkl"
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    with open(dataset_path, "rb") as f:
        tasks = pickle.load(f)
    
    # Create data loaders
    for task in tasks:
        task['train_loader'] = None  # Not needed for evaluation
        task['test_loader'] = torch.utils.data.DataLoader(
            task['test_data'], 
            batch_size=32, 
            shuffle=False, 
            num_workers=2
        )
    
    return tasks

def evaluate_model(model, tasks, device):
    """
    Evaluate model on all tasks
    """
    model.eval()
    results = {}
    
    # Store accuracy for each task when evaluated on all tasks
    # This gives us the complete continual learning performance matrix
    performance_matrix = np.zeros((len(tasks), len(tasks)))
    
    with torch.no_grad():
        for test_task_idx, test_task in enumerate(tasks):
            print(f"Evaluating on task {test_task_idx + 1}/{len(tasks)}...")
            
            correct = 0
            total = 0
            
            for pixel_values, labels in test_task['test_loader']:
                pixel_values = pixel_values.to(device)
                labels = labels.to(device)
                
                # Use the task_id for inference (paper assumes task identity known at inference)
                logits = model(pixel_values, task_id=test_task_idx)
                pred = logits.argmax(dim=1)
                correct += pred.eq(labels).sum().item()
                total += labels.size(0)
            
            accuracy = 100. * correct / total
            results[f'task_{test_task_idx}_accuracy'] = accuracy
            performance_matrix[test_task_idx, test_task_idx] = accuracy
            
            # Also evaluate on previous tasks to measure forgetting
            for prev_task_idx in range(test_task_idx):
                # Evaluate on previous task data with current model
                prev_task = tasks[prev_task_idx]
                prev_correct = 0
                prev_total = 0
                
                for pixel_values, labels in prev_task['test_loader']:
                    pixel_values = pixel_values.to(device)
                    labels = labels.to(device)
                    
                    logits = model(pixel_values, task_id=prev_task_idx)
                    pred = logits.argmax(dim=1)
                    prev_correct += pred.eq(labels).sum().item()
                    prev_total += labels.size(0)
                
                prev_accuracy = 100. * prev_correct / prev_total
                performance_matrix[test_task_idx, prev_task_idx] = prev_accuracy
    
    # Calculate metrics
    results['average_accuracy'] = np.mean([results[f'task_{i}_accuracy'] for i in range(len(tasks))])
    results['final_accuracy'] = results[f'task_{len(tasks)-1}_accuracy']
    
    # Calculate backward transfer (average accuracy on previous tasks after learning new ones)
    if len(tasks) > 1:
        backward_transfer = np.mean([performance_matrix[len(tasks)-1, i] for i in range(len(tasks)-1)])
        results['backward_transfer'] = backward_transfer
    else:
        results['backward_transfer'] = 0.0
    
    # Calculate forgetting (average drop in accuracy on previous tasks)
    if len(tasks) > 1:
        forgetting = 0
        for i in range(len(tasks) - 1):
            # Find the maximum accuracy achieved on task i
            max_acc_on_i = np.max(performance_matrix[:i+1, i])
            # Current accuracy on task i
            current_acc_on_i = performance_matrix[len(tasks)-1, i]
            forgetting += max_acc_on_i - current_acc_on_i
        forgetting /= (len(tasks) - 1)
        results['forgetting'] = forgetting
    else:
        results['forgetting'] = 0.0
    
    # Calculate forward transfer (average improvement on later tasks)
    if len(tasks) > 1:
        forward_transfer = 0
        for i in range(1, len(tasks)):
            # Compare with first task accuracy
            first_task_acc = performance_matrix[0, 0]
            current_task_acc = performance_matrix[i, i]
            forward_transfer += current_task_acc - first_task_acc
        forward_transfer /= (len(tasks) - 1)
        results['forward_transfer'] = forward_transfer
    else:
        results['forward_transfer'] = 0.0
    
    # Calculate average accuracy over all tasks
    results['average_accuracy_over_tasks'] = results['average_accuracy']
    
    return results, performance_matrix

def main():
    args = parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    print(f"Loading {args.dataset} dataset...")
    tasks = load_dataset(args.dataset)
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    
    # Initialize model with correct number of classes for first task
    num_classes_first_task = len(tasks[0]['classes'])
    model = SEMA(
        num_classes=num_classes_first_task,
        model_name="google/vit-base-patch16-224-in21k",
        adapter_dim=64,
        expansion_threshold=0.1,
        max_adapters_per_layer=3
    ).to(device)
    
    # Load checkpoint
    if args.model_path.endswith('.pt'):
        checkpoint = torch.load(args.model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        # Load model weights directly
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    
    print(f"Model loaded with {model.get_adapter_count()} adapters")
    
    # Evaluate model
    print("Evaluating model...")
    results, performance_matrix = evaluate_model(model, tasks, device)
    
    # Print results
    print("\n=== Evaluation Results ===")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    print(f"\nPerformance Matrix:")
    print(performance_matrix)
    
    # Save results to CSV
    os.makedirs(os.path.dirname(args.output_file) if os.path.dirname(args.output_file) else '.', exist_ok=True)
    
    with open(args.output_file, 'w') as f:
        f.write("Metric,Value\n")
        for key, value in results.items():
            if isinstance(value, float):
                f.write(f"{key},{value:.6f}\n")
            else:
                f.write(f"{key},{value}\n")
    
    # Save performance matrix
    matrix_file = args.output_file.replace('.csv', '_matrix.npy')
    np.save(matrix_file, performance_matrix)
    
    print(f"\nResults saved to {args.output_file}")
    print(f"Performance matrix saved to {matrix_file}")

if __name__ == "__main__":
    main()