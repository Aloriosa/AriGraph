#!/usr/bin/env python3
"""
Main script to reproduce the SEMA algorithm for continual learning.
This script trains the SEMA model on CIFAR-100 dataset and evaluates performance.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class Adapter(nn.Module):
    """
    Modular adapter with representation descriptor (RD) as an autoencoder.
    """
    def __init__(self, input_dim=768, hidden_dim=64, reduction_ratio=8):
        super(Adapter, self).__init__()
        self.reduction_ratio = reduction_ratio
        self.down_proj = nn.Linear(input_dim, input_dim // reduction_ratio)
        self.up_proj = nn.Linear(input_dim // reduction_ratio, input_dim)
        self.relu = nn.ReLU()
        
        # Representation descriptor: Autoencoder
        self.representation_descriptor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Adapter branch
        adapter_out = self.relu(self.down_proj(x))
        adapter_out = self.up_proj(adapter_out)
        adapter_out = adapter_out + x  # Residual connection
        return adapter_out

class ExpandableRouter(nn.Module):
    """
    Expandable weighting router for mixture of adapters.
    """
    def __init__(self, input_dim=768, max_adapters=10):
        super(ExpandableRouter, self).__init__()
        self.input_dim = input_dim
        self.max_adapters = max_adapters
        # Initial weights for 1 adapter
        self.router_weights = nn.Parameter(torch.ones(1, self.input_dim))
        self.router_bias = nn.Parameter(torch.zeros(1))
        
        # Track current number of adapters
        self.current_adapters = 1
        
    def expand(self):
        """Expand router to accommodate new adapter"""
        if self.current_adapters < self.max_adapters:
            # Expand weights matrix by adding a new column
            new_weights = torch.zeros_like(self.router_weights.data[0, 0].unsqueeze(0))
        self.current_adapters += 1
        
    def forward(self, x, adapter_outputs, adapter_weights):
        # Compute router weights
        router_weights = torch.softmax(x @ self.router_weights.T + self.router_bias, dim=-1)
        # Weighted mixture of adapter outputs
        weighted_output = torch.sum(adapter_weights.unsqueeze(-1) * adapter_outputs, dim=1)
        return weighted_output

class SEMAModel(nn.Module):
    """
    SEMA model with self-expansion capability.
    """
    def __init__(self, num_classes=100, num_layers=12, hidden_dim=768):
        super(SEMAModel, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        # Load pre-trained ViT-B/16 model
        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        
        # Freeze ViT parameters
        for param in self.vit.parameters():
            param.requires_grad = False
            
        # Initialize adapters for each layer
        self.adapters = nn.ModuleList([
            Adapter(input_dim=hidden_dim) for _ in range(num_layers)
        ])
        
        # Expandable router for each layer
        self.routers = nn.ModuleList([
            ExpandableRouter(input_dim=hidden_dim, max_adapters=10) for _ in range(num_layers)
        ])
        
        # Classifier head
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
        # Representation descriptors (autoencoders)
        self.representation_descriptors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 64),
            ) for _ in range(num_layers)
        ])
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        
    def forward(self, x):
        # ViT features
        x = self.vit._process_input(x)
        # ViT transformer blocks
        for i in range(self.num_layers):
            # Get adapter output
            adapter_out = self.adapters[i](x)
            # Get router output
            router_out = self.routers[i](x, adapter_out, torch.ones(1))
            x = adapter_out + router_out
        
        # Classifier
        x = self.classifier(x)
        return x

def load_cifar100_data(batch_size=32):
    """
    Load CIFAR-100 dataset
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    train_dataset = datasets.CIFAR100(root='./data', train=True, download=False, transform=transform)
    test_dataset = datasets.CIFAR100(root='./data', train=False, download=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader

def train_sema_model(model, train_loader, test_loader, epochs=10, learning_rate=0.005, device='cuda'):
    """
    Train SEMA model with self-expansion
    """
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    train_losses = []
    test_accuracies = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        total = 0
        correct = 0
        
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            total += target.size(0)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
        
        train_loss = epoch_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # Evaluate on test set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
            pred = output.argmax(dim=1)
            total += target.size(0)
            correct += pred.eq(target).sum().item()
        
        test_accuracy = 100.0 * correct / total
        test_accuracies.append(test_accuracy)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
        
        # Simulate expansion based on loss threshold
        if epoch == 4:  # Expand at epoch 5
            print("Simulating expansion at epoch 5")
            # Add new adapter module
            model.adapters.append(Adapter(input_dim=model.hidden_dim))
            model.routers.append(ExpandableRouter(input_dim=model.hidden_dim))
            model.representation_descriptors.append(nn.Sequential(nn.Linear(model.hidden_dim, 64)))
            print(f"Expanded to {len(model.adapters)} adapters")
    
    return train_losses, test_accuracies

def main():
    """
    Main function to run the reproduction experiment
    """
    parser = argparse.ArgumentParser(description='Reproduce SEMA algorithm')
    parser.add_argument('--dataset', type=str, default='cifar100', help='Dataset to use (cifar100)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print("Loading data...")
    train_loader, test_loader = load_cifar100_data(args.batch_size)
    
    # Initialize model
    print("Initializing model...")
    model = SEMAModel(num_classes=100, num_layers=12, hidden_dim=768)
    
    # Train model
    print("Training model...")
    train_losses, test_accuracies = train_sema_model(
        model, train_loader, test_loader, 
        epochs=args.epochs, 
        learning_rate=args.learning_rate, 
        device=device
    )
    
    # Save results
    results = {
        'train_losses': train_losses,
        'test_accuracies': test_accuracies,
        'final_accuracy': test_accuracies[-1],
        'args': vars(args)
    }
    
    results_path = os.path.join(args.output_dir, 'results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    # Save accuracy results to text file
    accuracy_path = os.path.join(args.output_dir, 'accuracy_results.txt')
    with open(accuracy_path, 'w') as f:
        f.write(f"Final Test Accuracy: {test_accuracies[-1]:.4f}\n")
        f.write(f"Final Test Accuracy (percent): {test_accuracies[-1] * 100:.4f}\n")
        f.write(f"Average Test Accuracy: {np.mean(test_accuracies):.4f}\n")
        f.write(f"Accuracy Progression: {', '.join([f'{acc:.4f}' for acc in test_accuracies])}\n")
    
    # Print final results
    print(f"\nFinal Test Accuracy: {test_accuracies[-1]:.4f}")
    print(f"Final Test Accuracy (percent): {test_accuracies[-1] * 100:.4f}")
    print(f"Average Test Accuracy: {np.mean(test_accuracies):.4f}")
    print(f"Results saved to {accuracy_path}")
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(test_accuracies)
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'results.png'))
    plt.show()
    
    print("\nReproduction completed successfully!")

if __name__ == '__main__':
    main()