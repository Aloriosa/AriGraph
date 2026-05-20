import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import time
import os
from torch.utils.data import DataLoader
from models.sema import SEMA
from models.vit import ViT
from data.data_loader import DataLoader
from utils.utils import get_device, set_seed, save_checkpoint, load_checkpoint, create_directory, get_reconstruction_error, compute_z_score

# Set seed for reproducibility
set_seed(42)

def train_sema(model, dataloader, optimizer, criterion, device, num_epochs=5):
    """
    Train SEMA model
    """
    model.train()
    total_loss = 0
    total_samples = 0
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(data)
        total_samples += len(data)
        
        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}/{len(dataloader)} Loss: {loss.item():.4f}")
    
    return total_loss / total_samples

def evaluate_sema(model, dataloader, criterion, device):
    """
    Evaluate SEMA model
    """
    model.eval()
    correct = 0
    total = 0
    loss = 0
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += len(target)
    
    return correct / total, loss / len(dataloader)

def main():
    """
    Main function to run the reproduction
    """
    parser = argparse.ArgumentParser(description='Reproduce SEMA paper results')
    parser.add_argument('--dataset', type=str, default='cifar100', help='Dataset to use')
    parser.add_argument('--batch-size', type=int, default=32, help='Input batch size')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint to load')
    parser.add_argument('--test', action='store_true', help='Test mode')
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create directories
    create_directory('results')
    
    # Initialize data loader
    data_loader = DataLoader(dataset_name=args.dataset, data_dir='./data', batch_size=args.batch_size)
    
    # Setup tasks
    data_loader.setup_tasks(num_tasks=20)
    
    # Initialize model
    model = SEMA(input_dim=768, num_layers=12, hidden_dim=64)
    model = model.to(device)
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Initialize loss function
    criterion = nn.CrossEntropyLoss()
    
    # Load checkpoint if provided
    start_epoch = 0
    if args.checkpoint:
        start_epoch = load_checkpoint(model, optimizer, args.checkpoint)
    
    # Training loop
    print("Starting training...")
    for epoch in range(start_epoch, args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        
        # Get current task
        current_task_classes = data_loader.get_current_task()
        train_loader = data_loader.get_train_loader(current_task_classes)
        
        # Train
        train_loss = train_sema(model, train_loader, optimizer, criterion, device, num_epochs=1)
        
        # Evaluate
        test_loader = data_loader.get_test_loader(current_task_classes)
        test_acc, test_loss = evaluate_sema(model, test_loader, criterion, device)
        
        print(f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Test Acc: {test_acc:.4f}")
        
        # Save checkpoint
        save_checkpoint(model, optimizer, epoch, f'results/checkpoint_epoch_{epoch}.pth")
    
    # Final evaluation
    print("Final evaluation...")
    final_acc = []
    for task_idx in range(len(data_loader.tasks)):
        task_classes = data_loader.tasks[task_idx]
        test_loader = data_loader.get_test_loader(task_classes)
        acc, _ = evaluate_sema(model, test_loader, criterion, device)
        final_acc.append(acc)
        print(f"Task {task_idx}: Acc: {acc:.4f}")
    
    # Save final model
    save_checkpoint(model, optimizer, args.epochs, 'results/final_model.pth")
    
    # Print summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Dataset: {args.dataset}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Final average accuracy: {np.mean(final_acc):.4f}")
    print(f"Final accuracy per task: {[f'{a:.4f}' for a in final_acc]}")
    print(f"Final standard deviation: {np.std(final_acc):.4f}")
    print("="*50)
    
    # Create output.csv file
    with open('output.csv', 'w') as f:
        f.write("metric,value\n")
        f.write(f"final_average_accuracy,{np.mean(final_acc)}\n")
        f.write(f"final_accuracy_std,{np.std(final_acc)}\n")
        f.write(f"final_accuracy_per_task,{';'.join([f'{a:.4f}' for a in final_acc])}\n")
    
    print("\nResults saved to output.csv")

if __name__ == "__main__":
    main()