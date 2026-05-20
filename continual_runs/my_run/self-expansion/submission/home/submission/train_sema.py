#!/usr/bin/env python3
"""
Train SEMA model for continual learning
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pickle
import numpy as np
from tqdm import tqdm
import wandb
from sema_model import SEMA
import json

def parse_args():
    parser = argparse.ArgumentParser(description='Train SEMA model for continual learning')
    parser.add_argument('--dataset', type=str, default='split_cifar100', 
                       choices=['split_cifar100', 'split_tiny_imagenet'],
                       help='Dataset to use')
    parser.add_argument('--model', type=str, default='vit_b_16',
                       help='Model type (currently only vit_b_16 supported)')
    parser.add_argument('--num_tasks', type=int, default=10,
                       help='Number of tasks')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--adapter_dim', type=int, default=64,
                       help='Adapter dimension')
    parser.add_argument('--expansion_threshold', type=float, default=0.1,
                       help='Distribution shift detection threshold')
    parser.add_argument('--max_adapters_per_layer', type=int, default=3,
                       help='Maximum adapters per layer')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs per task')
    parser.add_argument('--output_dir', type=str, default='./models/sema',
                       help='Output directory for models')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases for logging')
    
    return parser.parse_args()

def train_task(model, dataloader, optimizer, device, epoch, task_id, num_epochs):
    """
    Train model for one task
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc=f'Task {task_id}, Epoch {epoch}/{num_epochs}')
    
    for batch_idx, (pixel_values, labels) in enumerate(progress_bar):
        pixel_values = pixel_values.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(pixel_values, task_id=task_id)
        loss = nn.CrossEntropyLoss()(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)
        
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy

def evaluate_model(model, dataloader, device, task_id, all_tasks_data):
    """
    Evaluate model on all tasks
    """
    model.eval()
    results = {}
    
    with torch.no_grad():
        for task_idx, task_data in enumerate(all_tasks_data):
            correct = 0
            total = 0
            
            for pixel_values, labels in task_data['test_loader']:
                pixel_values = pixel_values.to(device)
                labels = labels.to(device)
                
                logits = model(pixel_values, task_id=task_idx)
                pred = logits.argmax(dim=1)
                correct += pred.eq(labels).sum().item()
                total += labels.size(0)
            
            accuracy = 100. * correct / total
            results[f'task_{task_idx}_accuracy'] = accuracy
            
            if task_idx == task_id:
                results[f'current_task_accuracy'] = accuracy
    
    return results

def main():
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize W&B
    if args.use_wandb:
        wandb.init(
            project="sema-continual-learning",
            name=f"sema_{args.dataset}_{args.num_tasks}_tasks",
            config=args
        )
    
    # Load dataset
    print(f"Loading {args.dataset} dataset...")
    dataset_path = f"./data/{args.dataset}_tasks.pkl"
    
    if not os.path.exists(dataset_path):
        print(f"Dataset not found: {dataset_path}")
        print("Please run prepare_datasets.py first")
        return
    
    with open(dataset_path, "rb") as f:
        tasks = pickle.load(f)
    
    # Create data loaders
    for task in tasks:
        task['train_loader'] = DataLoader(
            task['train_data'], 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=2
        )
        task['test_loader'] = DataLoader(
            task['test_data'], 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=2
        )
    
    # Initialize SEMA model
    # Get number of classes for first task
    num_classes_first_task = len(tasks[0]['classes'])
    model = SEMA(
        num_classes=num_classes_first_task,
        model_name="google/vit-base-patch16-224-in21k",
        adapter_dim=args.adapter_dim,
        expansion_threshold=args.expansion_threshold,
        max_adapters_per_layer=args.max_adapters_per_layer
    ).to(device)
    
    print(f"Model initialized with {model.get_adapter_count()} adapters")
    
    # Training loop
    all_results = []
    task_accuracies = []
    
    for task_id in range(args.num_tasks):
        print(f"\n=== Training Task {task_id + 1}/{args.num_tasks} ===")
        
        # Update classifier if needed
        if task_id > 0:
            model.add_task(len(tasks[task_id]['classes']))
        
        # Create optimizer for this task
        # Only optimize adapter and classifier parameters
        optimizer = optim.Adam(
            [
                {'params': model.adapters.parameters(), 'lr': args.learning_rate},
                {'params': model.representation_descriptors.parameters(), 'lr': args.learning_rate},
                {'params': model.routers.parameters(), 'lr': args.learning_rate},
                {'params': model.classifier.parameters(), 'lr': args.learning_rate}
            ],
            lr=args.learning_rate,
            betas=(0.9, 0.999)
        )
        
        # Train for specified epochs
        for epoch in range(args.epochs):
            avg_loss, accuracy = train_task(
                model, tasks[task_id]['train_loader'], optimizer, device, epoch + 1, task_id, args.epochs
            )
            
            if args.use_wandb:
                wandb.log({
                    f'task_{task_id}/epoch': epoch + 1,
                    f'task_{task_id}/loss': avg_loss,
                    f'task_{task_id}/accuracy': accuracy
                })
        
        # Evaluate on all tasks
        results = evaluate_model(model, None, device, task_id, tasks)
        all_results.append(results)
        
        # Log current task accuracy
        current_task_acc = results[f'current_task_accuracy']
        task_accuracies.append(current_task_acc)
        
        print(f"Task {task_id + 1} training completed. Accuracy: {current_task_acc:.2f}%")
        print(f"Total adapters: {model.get_adapter_count()}")
        
        # Save model checkpoint
        os.makedirs(args.output_dir, exist_ok=True)
        checkpoint_path = os.path.join(args.output_dir, f"sema_task_{task_id}.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'task_id': task_id,
            'adapter_expansion_history': model.get_expansion_history(),
            'task_accuracies': task_accuracies
        }, checkpoint_path)
        
        print(f"Model checkpoint saved to {checkpoint_path}")
    
    # Save final results
    results_path = os.path.join(args.output_dir, "training_results.json")
    with open(results_path, "w") as f:
        json.dump({
            'task_accuracies': task_accuracies,
            'all_results': all_results,
            'adapter_expansion_history': model.get_expansion_history(),
            'total_adapters': model.get_adapter_count()
        }, f, indent=2)
    
    print(f"\nTraining completed!")
    print(f"Final adapter count: {model.get_adapter_count()}")
    print(f"Task accuracies: {task_accuracies}")
    print(f"Average accuracy: {np.mean(task_accuracies):.2f}%")
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, "sema_final.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()