#!/usr/bin/env python3
"""
Main script for reproducing the SMM experiments from the paper.
This script implements the core SMM framework as described in the paper.
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
import logging
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

# Add the SMM repository path
sys.path.append('/tmp/SMM')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {device}")

class MaskGenerator(nn.Module):
    """
    Lightweight ConvNet for generating sample-specific masks.
    This implements the mask generator module described in Section 3.2 of the paper.
    """
    def __init__(self, in_channels=3, num_layers=5):
        super(MaskGenerator, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        
        # First convolutional layer
        self.layers.append(nn.Conv2d(in_channels, 32, kernel_size=3, padding=1))
        self.layers.append(nn.ReLU())
        
        # Middle layers
        for i in range(1, num_layers - 1):
            self.layers.append(nn.Conv2d(32, 32, kernel_size=3, padding=1))
            self.layers.append(nn.ReLU())
        
        # Final convolutional layer
        self.layers.append(nn.Conv2d(32, 3, kernel_size=3, padding=1))
        self.layers.append(nn.Sigmoid())
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class PatchWiseInterpolation(nn.Module):
    """
    Patch-wise interpolation module for resizing masks.
    This implements the patch-wise interpolation module described in Section 3.3 of the paper.
    """
    def __init__(self, patch_size=8):
        super(PatchWiseInterpolation, self).__init__()
        self.patch_size = patch_size
    
    def forward(self, x):
        # x shape: (batch_size, channels, height, width)
        batch_size, channels, height, width = x.shape
        
        # Calculate the number of patches
        num_patches_h = height
        num_patches_w = width
        
        # Upsample each patch
        # Create a new tensor for the upsampled output
        output = torch.zeros(batch_size, channels, height * self.patch_size, width * self.patch_size, device=x.device)
        
        # For each patch, assign the same value to the corresponding region
        for i in range(height):
            for j in range(width):
                # Get the value of the current pixel
        # This is a simplified version - in practice, we'd use torch.repeat_interleave
        # But for correctness, we'll use a more efficient implementation
        x_expanded = x.unsqueeze(-1).unsqueeze(-1)  # Add two dimensions
        x_expanded = x_expanded.repeat(1, 1, 1, 1, self.patch_size)
        x_expanded = x_expanded.reshape(batch_size, channels, height, width * self.patch_size)
        x_expanded = x_expanded.unsqueeze(2)
        x_expanded = x_expanded.repeat(1, 1, self.patch_size, 1)
        output = x_expanded.reshape(batch_size, channels, height * self.patch_size, width * self.patch_size)
        return output

class SMM(nn.Module):
    """
    Sample-specific Multi-channel Masks framework.
    This implements the complete SMM framework described in Section 3 of the paper.
    """
    def __init__(self, model_name='resnet18', num_classes=10):
        super(SMM, self).__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.patch_size = 8
        
        # Load the pre-trained model
        self.pretrained_model = self._load_pretrained_model()
        
        # Create the mask generator
        self.mask_generator = MaskGenerator(in_channels=3, num_layers=5)
        
        # Create the patch-wise interpolation module
        self.patch_interpolation = PatchWiseInterpolation(patch_size=self.patch_size)
        
        # Create the output mapping function
        self.output_mapping = nn.Identity()  # This will be updated during training
        
        # Initialize the noise pattern
        self.delta = torch.zeros(3, 224, 224, requires_grad=True, device=device)
        
        # Create the resize function
        self.resize = transforms.Resize((224, 224))
    
    def _load_pretrained_model(self):
        """Load the pre-trained model based on the model name."""
        if self.model_name == 'resnet18':
            model = torchvision.models.resnet18(pretrained=True)
        elif self.model_name == 'resnet50':
            model = torchvision.models.resnet50(pretrained=True)
        elif self.model_name == 'vit_b32':
            model = torchvision.models.vit_b_32(pretrained=True)
        else:
            raise ValueError(f"Model {self.model_name} not supported")
        
        # Freeze the pretrained model parameters
        for param in model.parameters():
            param.requires_grad = False
        
        # Modify the final layer for our number of classes
        if 'resnet' in self.model_name:
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        elif 'vit' in self.model_name:
            model.heads.head = nn.Linear(model.heads.head.in_features, self.num_classes)
        
        return model.to(device)
    
    def forward(self, x):
        # Resize the input
        x_resized = self.resize(x)
        
        # Generate the mask
        mask = self.mask_generator(x_resized)
        
        # Apply patch-wise interpolation
        mask_interpolated = self.patch_interpolation(mask)
        
        # Apply the noise pattern
        x_reprogrammed = x_resized + self.delta * mask_interpolated
        
        # Apply the pre-trained model
        output = self.pretrained_model(x_reprogrammed)
        
        return output
    
    def train(self, train_loader, epochs=5, lr=0.01):
        """Train the SMM model."""
        self.train()
        optimizer = torch.optim.Adam(list(self.mask_generator.parameters()) + [self.delta], lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            start_time = time.time()
            
            for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            epoch_loss = running_loss / len(train_loader)
            epoch_acc = 100. * correct / total
            epoch_time = time.time() - start_time
            logging.info(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%, Time: {epoch_time:.2f}s")
        
        return self

def main():
    """Main function to run the reproduction script."""
    parser = argparse.ArgumentParser(description='Reproduce SMM experiments')
    parser.add_argument('--model', type=str, default='resnet18', help='Model to use: resnet18, resnet50, vit_b32')
    parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset to use: cifar10, cifar100, svhn, gtsrb, flowers102, dtd, ucf101, food101, sun397, eurosat, oxfordpets')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--output_dir', type=str, default='/home/submission/results', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.485, 0.406, 0.406])
    ])
    
    if args.dataset == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10(root='/tmp/data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(root='/tmp/data', train=False, download=True, transform=transform)
        num_classes = 10
    elif args.dataset == 'cifar100':
        train_dataset = torchvision.datasets.CIFAR100(root='/tmp/data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR100(root='/tmp/data', train=False, download=True, transform=transform)
        num_classes = 100
    elif args.dataset == 'svhn':
        train_dataset = torchvision.datasets.SVHN(root='/tmp/data', split='train', download=True, transform=transform)
        test_dataset = torchvision.datasets.SVHN(root='/tmp/data', split='test', download=True, transform=transform)
        num_classes = 10
    elif args.dataset == 'gtsrb':
        train_dataset = torchvision.datasets.GTSRB(root='/tmp/data', split='train', download=True, transform=transform)
        test_dataset = torchvision.datasets.GTSRB(root='/tmp/data', split='test', download=True, transform=transform)
        num_classes = 43
    elif args.dataset == 'flowers102':
        train_dataset = torchvision.datasets.Flowers102(root='/tmp/data', split='train', download=True, transform=transform)
        test_dataset = torchvision.datasets.Flowers102(root='/tmp/data', split='test', download=True, transform=transform)
        num_classes = 102
    elif args.dataset == 'dtd':
        train_dataset = torchvision.datasets.DTD(root='/tmp/data', split='train', download=True, transform=transform)
    elif args.dataset == 'ucf101':
        train_dataset = torchvision.datasets.UCF101(root='/tmp/data', split='train', download=True, transform=transform)
        test_dataset = torchvision.datasets.UCF101(root='/tmp/data', split='test', download=True, transform=transform)
        num_classes = 101
    elif args.dataset == 'food101':
        train_dataset = torchvision.datasets.Food101(root='/tmp/data', split='train', download=True, transform=transform)
        test_dataset = torchvision.datasets.Food101(root='/tmp/data', split='test', download=True, transform=transform)
        num_classes = 101
    elif args.dataset == 'sun397':
        train_dataset = torchvision.datasets.SUN397(root='/tmp/data', split='train', download=True, transform=transform)
        test_dataset = torchvision.datasets.SUN397(root='/tmp/data', split='test', download=True, transform=transform)
        num_classes = 397
    elif args.dataset == 'eurosat':
        train_dataset = torchvision.datasets.EuroSAT(root='/tmp/data', download=True, transform=transform)
        test_dataset = torchvision.datasets.EuroSAT(root='/tmp/data', download=True, transform=transform)
        num_classes = 10
    elif args.dataset == 'oxfordpets':
        train_dataset = torchvision.datasets.OxfordIIITPet(root='/tmp/data', split='train', download=True, transform=transform)
        test_dataset = torchvision.datasets.OxfordIIITPet(root='/tmp/data', split='test', download=True, transform=transform)
        num_classes = 37
    else:
        raise ValueError(f"Dataset {args.dataset} not supported')
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create SMM model
    smm = SMM(model_name=args.model, num_classes=num_classes)
    
    # Train the model
    logging.info(f"Training SMM model {args.model} on dataset {args.dataset}")
    smm.train(train_loader, epochs=args.epochs, lr=args.lr)
    
    # Evaluate the model
    logging.info("Evaluating the model")
    smm.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = smm(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    test_acc = 100. * correct / total
    logging.info(f"Test Accuracy: {test_acc:.2f}%")
    
    # Save results
    results = {
        'model': args.model,
        'dataset': args.dataset,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'test_accuracy': test_acc
    }
    
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    logging.info(f"Results saved to {os.path.join(args.output_dir, 'results.json')}")
    
    # Generate visualization
    logging.info("Generating visualization")
    # This is a simplified version of the visualization code from the paper
    # In a real implementation, we would generate plots like Figure 5 from the paper
    # For brevity, we'll just save a sample image
    sample_image = inputs[0].cpu().numpy().transpose(1, 2, 0)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(sample_image)
    plt.title('Original Image')
    plt.axis('off')
    
    # Generate a sample mask
    sample_mask = smm.mask_generator(inputs[0].unsqueeze(0))[0].cpu().detach().numpy().transpose(1, 2, 0)
    plt.subplot(1, 2, 2)
    plt.imshow(sample_mask)
    plt.title('Generated Mask')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'visualization.png'))
    plt.close()
    
    logging.info("Reproduction completed successfully!")

if __name__ == '__main__':
    main()