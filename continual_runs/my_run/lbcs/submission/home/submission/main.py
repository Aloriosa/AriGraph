import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader
from tqdm import tqdm
import pickle
import json
from collections import defaultdict

# Set random seeds for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class LeNet(nn.Module):
    """Simplified LeNet architecture for MNIST/FMNIST and CIFAR-10"""
    def __init__(self, num_classes=10, in_channels=1):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, 5)
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class GradientImportanceScorer:
    """Computes gradient-based importance scores for data points"""
    def __init__(self, model, criterion, device):
        self.model = model
        self.criterion = criterion
        self.device = device
        
    def compute_scores(self, dataloader):
        """Compute importance scores based on gradient norms"""
        self.model.eval()
        scores = []
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                output = self.model(data)
                loss = self.criterion(output, target)
                
                # Compute gradients
                self.model.zero_grad()
                loss.backward(retain_graph=True)
                
                # Compute gradient norm for each sample
                for param in self.model.parameters():
                    if param.grad is not None:
                        # Compute per-sample gradient norms
                        grad_norms = torch.norm(param.grad.view(param.grad.size(0), -1), dim=1)
                        scores.extend(grad_norms.cpu().numpy())
                        break  # Use first parameter's gradient as proxy for all
                
                # If no gradients (e.g., batch norm), use loss as proxy
                if len(scores) == 0:
                    scores.extend(loss.cpu().numpy())
        
        # Normalize scores
        scores = np.array(scores)
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        return scores

class LBCS:
    """Lexicographic Bilevel Coreset Selection algorithm"""
    
    def __init__(self, model, criterion, device, coreset_size, performance_threshold=0.95, 
                 inner_epochs=10, outer_iterations=5, batch_size=64):
        self.model = model
        self.criterion = criterion
        self.device = device
        self.coreset_size = coreset_size
        self.performance_threshold = performance_threshold
        self.inner_epochs = inner_epochs
        self.outer_iterations = outer_iterations
        self.batch_size = batch_size
        
    def select_coreset(self, full_dataset, val_dataset, test_dataset):
        """Main LBCS algorithm: lexicographic optimization of performance then size"""
        # Step 1: Train on full dataset to get baseline performance
        print("Training on full dataset to establish baseline...")
        baseline_model = self._train_model(full_dataset, epochs=10)
        baseline_acc = self._evaluate_model(baseline_model, test_dataset)
        print(f"Baseline accuracy on test set: {baseline_acc:.4f}")
        
        # Define performance constraint (must not degrade below threshold of baseline)
        performance_constraint = baseline_acc * self.performance_threshold
        print(f"Performance constraint: {performance_constraint:.4f}")
        
        # Step 2: Compute gradient importance scores for all data points
        print("Computing gradient importance scores...")
        scorer = GradientImportanceScorer(self.model, self.criterion, self.device)
        full_loader = DataLoader(full_dataset, batch_size=self.batch_size, shuffle=False)
        scores = scorer.compute_scores(full_loader)
        
        # Sort data points by importance scores (descending)
        indices = np.argsort(scores)[::-1]  # Highest importance first
        
        # Step 3: Iterative coreset selection with performance constraint
        # Start with empty coreset and gradually add points until constraint is met
        coreset_indices = []
        current_coreset_size = 0
        best_coreset = None
        best_accuracy = 0.0
        
        # We'll try different coreset sizes from small to large
        # But we want minimal size that meets constraint
        # So we'll start with smallest possible and increase until constraint met
        for size in range(1, min(self.coreset_size + 1, len(full_dataset) + 1)):
            # Select top 'size' points by importance
            current_coreset_indices = indices[:size]
            current_coreset = Subset(full_dataset, current_coreset_indices)
            
            # Train model on current coreset
            print(f"Training on coreset of size {size}...")
            coreset_model = self._train_model(current_coreset, epochs=self.inner_epochs)
            
            # Evaluate on validation set
            val_acc = self._evaluate_model(coreset_model, val_dataset)
            print(f"Validation accuracy: {val_acc:.4f}")
            
            # Check if performance constraint is met
            if val_acc >= performance_constraint:
                # We found a coreset that meets the constraint
                # But we want the minimal one, so continue to see if smaller works
                # However, since we're increasing size, this is the first valid one
                best_coreset = current_coreset_indices
                best_accuracy = val_acc
                current_coreset_size = size
                print(f"Found minimal coreset of size {size} with validation accuracy {val_acc:.4f}")
                break
        
        # If we didn't find one with the constraint, use the largest possible
        if best_coreset is None:
            print("No coreset met performance constraint, using largest possible")
            best_coreset = indices[:self.coreset_size]
            best_accuracy = self._evaluate_model(self._train_model(Subset(full_dataset, best_coreset), epochs=self.inner_epochs), val_dataset)
            current_coreset_size = len(best_coreset)
        
        # Final training on best coreset
        final_model = self._train_model(Subset(full_dataset, best_coreset), epochs=self.inner_epochs)
        
        # Evaluate on test set
        test_acc = self._evaluate_model(final_model, test_dataset)
        
        return {
            'coreset_indices': best_coreset,
            'coreset_size': current_coreset_size,
            'validation_accuracy': best_accuracy,
            'test_accuracy': test_acc,
            'baseline_accuracy': baseline_acc,
            'performance_constraint': performance_constraint,
            'model': final_model
        }
    
    def _train_model(self, dataset, epochs=10):
        """Train model on given dataset"""
        model = LeNet(num_classes=10, in_channels=1 if dataset.dataset.__class__.__name__ == 'FashionMNIST' else 3)
        model.to(self.device)
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0
            total = 0
            
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
            
            avg_loss = total_loss / len(dataloader)
            acc = 100. * correct / total
            if epoch % 5 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Acc: {acc:.2f}%")
        
        return model
    
    def _evaluate_model(self, model, dataset):
        """Evaluate model on dataset"""
        model.eval()
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        return correct / total

def load_dataset(dataset_name, data_dir='./data'):
    """Load and preprocess dataset"""
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) if dataset_name == 'cifar10' or dataset_name == 'svhn' else transforms.Normalize((0.2860,), (0.3530,))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) if dataset_name == 'cifar10' or dataset_name == 'svhn' else transforms.Normalize((0.2860,), (0.3530,))
    ])
    
    if dataset_name == 'fmnist':
        train_dataset = torchvision.datasets.FashionMNIST(
            root=data_dir, train=True, download=True, transform=transform_train)
        test_dataset = torchvision.datasets.FashionMNIST(
            root=data_dir, train=False, download=True, transform=transform_test)
        # Create validation set from training data
        train_size = len(train_dataset)
        val_size = int(0.1 * train_size)
        train_size = train_size - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size])
        
    elif dataset_name == 'svhn':
        train_dataset = torchvision.datasets.SVHN(
            root=data_dir, split='train', download=True, transform=transform_train)
        test_dataset = torchvision.datasets.SVHN(
            root=data_dir, split='test', download=True, transform=transform_test)
        # Create validation set from training data
        train_size = len(train_dataset)
        val_size = int(0.1 * train_size)
        train_size = train_size - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size])
        
    elif dataset_name == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=transform_test)
        # Create validation set from training data
        train_size = len(train_dataset)
        val_size = int(0.1 * train_size)
        train_size = train_size - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size])
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return train_dataset, val_dataset, test_dataset

def main():
    parser = argparse.ArgumentParser(description='Lexicographic Bilevel Coreset Selection')
    parser.add_argument('--dataset', type=str, default='fmnist', choices=['fmnist', 'svhn', 'cifar10'],
                        help='Dataset to use')
    parser.add_argument('--coreset_size', type=int, default=1000,
                        help='Maximum coreset size (will find minimal size that meets constraint)')
    parser.add_argument('--performance_threshold', type=float, default=0.95,
                        help='Threshold for performance constraint (fraction of baseline)')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--inner_epochs', type=int, default=10,
                        help='Number of epochs for inner loop training')
    parser.add_argument('--outer_iterations', type=int, default=5,
                        help='Number of outer iterations')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    print(f"Loading {args.dataset} dataset...")
    train_dataset, val_dataset, test_dataset = load_dataset(args.dataset)
    
    # Initialize LBCS
    model = LeNet(num_classes=10, in_channels=1 if args.dataset == 'fmnist' else 3)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    lbcs = LBCS(
        model=model,
        criterion=criterion,
        device=device,
        coreset_size=args.coreset_size,
        performance_threshold=args.performance_threshold,
        inner_epochs=args.inner_epochs,
        outer_iterations=args.outer_iterations,
        batch_size=args.batch_size
    )
    
    # Perform coreset selection
    print(f"Starting LBCS coreset selection for {args.dataset}...")
    result = lbcs.select_coreset(train_dataset, val_dataset, test_dataset)
    
    # Save results
    os.makedirs(args.output, exist_ok=True)
    
    # Save coreset indices
    with open(os.path.join(args.output, 'coreset_indices.pkl'), 'wb') as f:
        pickle.dump(result['coreset_indices'], f)
    
    # Save model
    torch.save(result['model'].state_dict(), os.path.join(args.output, 'model.pth'))
    
    # Save results as JSON
    result_summary = {
        'dataset': args.dataset,
        'coreset_size': result['coreset_size'],
        'validation_accuracy': float(result['validation_accuracy']),
        'test_accuracy': float(result['test_accuracy']),
        'baseline_accuracy': float(result['baseline_accuracy']),
        'performance_constraint': float(result['performance_constraint']),
        'seed': args.seed,
        'coreset_size_requested': args.coreset_size,
        'performance_threshold': args.performance_threshold
    }
    
    with open(os.path.join(args.output, 'results.json'), 'w') as f:
        json.dump(result_summary, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("LBCS RESULTS SUMMARY")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Requested coreset size: {args.coreset_size}")
    print(f"Actual coreset size: {result['coreset_size']}")
    print(f"Baseline accuracy: {result['baseline_accuracy']:.4f}")
    print(f"Performance constraint: {result['performance_constraint']:.4f}")
    print(f"Validation accuracy: {result['validation_accuracy']:.4f}")
    print(f"Test accuracy: {result['test_accuracy']:.4f}")
    print(f"Accuracy relative to baseline: {result['test_accuracy']/result['baseline_accuracy']:.4f}")
    print("="*60)
    
    # Save detailed results
    with open(os.path.join(args.output, 'detailed_results.txt'), 'w') as f:
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Requested coreset size: {args.coreset_size}\n")
        f.write(f"Actual coreset size: {result['coreset_size']}\n")
        f.write(f"Baseline accuracy: {result['baseline_accuracy']:.6f}\n")
        f.write(f"Performance constraint: {result['performance_constraint']:.6f}\n")
        f.write(f"Validation accuracy: {result['validation_accuracy']:.6f}\n")
        f.write(f"Test accuracy: {result['test_accuracy']:.6f}\n")
        f.write(f"Accuracy relative to baseline: {result['test_accuracy']/result['baseline_accuracy']:.6f}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Performance threshold: {args.performance_threshold}\n")
    
    print(f"Results saved to {args.output}")

if __name__ == '__main__':
    main()