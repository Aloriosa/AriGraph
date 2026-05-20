#!/usr/bin/env python3
"""
Main script for reproducing the SMM paper results.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class LightweightMaskGenerator(nn.Module):
    """
    Lightweight CNN for generating sample-specific three-channel masks.
    This implements the mask generator from Section 3.2 of the paper.
    """
    def __init__(self, input_channels=3, num_layers=5):
        super(LightweightMaskGenerator, self).__init__()
        
        layers = []
        in_channels = input_channels
        
        # Create 5-layer CNN architecture (Figure 8 in Appendix A.2)
        for i in range(num_layers):
            # Use 3x3 convolutions with padding=1 to maintain spatial dimensions
            layers.append(nn.Conv2d(in_channels, 64, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            in_channels = 64
            
            # Add MaxPooling every other layer (3 total pooling layers)
            if i < 3:  # Only add pooling for first 3 layers
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Final layer outputs 3 channels (as specified in Section 3.2)
        layers.append(nn.Conv2d(64, 3, kernel_size=3, padding=1))
        
        self.model = nn.Sequential(*layers)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return torch.sigmoid(self.model(x))  # Sigmoid to get values between 0-1

class PatchWiseInterpolation(nn.Module):
    """
    Patch-wise interpolation module to resize masks to match pattern dimensions.
    Implements Section 3.3 of the paper.
    """
    def __init__(self, patch_size=8):
        super(PatchWiseInterpolation, self).__init__()
        self.patch_size = patch_size
    
    def forward(self, masks):
        """
        Upsample masks using patch-wise interpolation.
        This avoids complex gradient computations and simplifies training.
        """
        # Get original size
        batch_size, channels, h, w = masks.shape
        
        # Calculate new size after upsampling
        new_h = h * self.patch_size
        new_w = w * self.patch_size
        
        # Use nearest neighbor interpolation to upsample
        # This is simpler than bilinear/bicubic interpolation
        upsampled = torch.nn.functional.interpolate(
            masks, 
            size=(new_h, new_w), 
            mode='nearest'
        )
        
        return upsampled

class SMM(nn.Module):
    """
    Sample-specific Multi-channel Masks (SMM) framework.
    Implements the complete SMM framework from Section 3.1 of the paper.
    """
    def __init__(self, input_size=224, patch_size=8, num_layers=5):
        super(SMM, self).__init__()
        
        self.input_size = input_size
        self.patch_size = patch_size
        self.mask_generator = LightweightMaskGenerator(input_channels=3, num_layers=num_layers)
        self.interpolation = PatchWiseInterpolation(patch_size=patch_size)
        
        # Learnable noise pattern
        self.delta = nn.Parameter(torch.zeros(3, input_size, input_size))
        
        # Initialize delta
        nn.init.normal_(self.delta, mean=0.0, std=0.01)
    
    def forward(self, x):
        # Generate sample-specific mask
        mask = self.mask_generator(x)
        
        # Interpolate to match pattern size
        interpolated_mask = self.interpolation(mask)
        
        # Apply noise pattern
        x_reprogrammed = x + interpolated_mask * self.delta
        
        return x_reprogrammed

class ReprogrammedClassifier(nn.Module):
    """
    Wrapper for pre-trained classifier with reprogramming.
    """
    def __init__(self, model_name='resnet18', num_classes=10):
        super(ReprogrammedClassifier, self).__init__()
        
        # Load pre-trained model
        if model_name == 'resnet18':
            self.backbone = torchvision.models.resnet18(pretrained=True)
        elif model_name == 'resnet50':
            self.backbone = torchvision.models.resnet50(pretrained=True)
        elif model_name == 'vit':
            self.backbone = torchvision.models.vit_b_32(pretrained=True)
        
        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Replace final layer
        if model_name in ['resnet18', 'resnet50']:
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)
        else:
            in_features = self.backbone.heads.head.in_features
            self.backbone.heads.head = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)

def load_dataset(dataset_name, batch_size=64):
    """
    Load datasets as specified in the paper (CIFAR10, CIFAR100, SVHN, etc.)
    """
    if dataset_name == 'cifar10':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2465, 0.247, 0.2465))
        ])
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        num_classes = 10
    elif dataset_name == 'cifar100':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2031, 0.2015, 0.2015))
        ])
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        num_classes = 100
    elif dataset_name == 'svhn':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4308, 0.4308, 0.4308), (0.4308, 0.4308, 0.4308))
        ])
        train_dataset = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform)
        test_dataset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform)
        num_classes = 10
    elif dataset_name == 'gtsrb':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.3403, 0.3403, 0.3403), (0.3403, 0.3403, 0.3403))
        ])
        train_dataset = torchvision.datasets.GTSRB(root='./data', split='train', download=True, transform=transform)
        test_dataset = torchvision.datasets.GTSRB(root='./data', split='test', download=True, transform=transform)
        num_classes = 43
    elif dataset_name == 'flowers102':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4308, 0.4308, 0.4308), (0.4308, 0.4308, 0.4308))
        ])
        train_dataset = torchvision.datasets.Flowers102(root='./data', split='train', download=True, transform=transform)
        test_dataset = torchvision.datasets.Flowers102(root='./data', split='test', download=True, transform=transform)
        num_classes = 102
    elif dataset_name == 'dtd':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4308, 0.4308, 0.4308), (0.4303, 0.4303, 0.4303))
        ])
        train_dataset = torchvision.datasets.DTD(root='./data', split='train', download=True, transform=transform)
        test_dataset = torchvision.datasets.DTD(root='./data', split='test', download=True, transform=transform)
        num_classes = 47
    elif dataset_name == 'ucf101':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4308, 0.4308, 0.4308), (0.4303, 0.4303, 0.4303))
        ])
        train_dataset = torchvision.datasets.UCF101(root='./data', split='train', download=True, transform=transform)
        test_dataset = torchvision.datasets.UCF101(root='./data', split='test', download=True, transform=transform)
        num_classes = 101
    elif dataset_name == 'food101':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4308, 0.4308, 0.4308), (0.4303, 0.4303, 0.4303))
        ])
        train_dataset = torchvision.datasets.Food101(root='./data', split='train', download=True, transform=transform)
        test_dataset = torchvision.datasets.Food101(root='./data', split='test', download=True, transform=transform)
        num_classes = 101
    elif dataset_name == 'sun397':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4308, 0.4308, 0.4308), (0.4303, 0.4303, 0.4303))
        ])
        train_dataset = torchvision.datasets.SUN397(root='./data', split='train', download=True, transform=transform)
        test_dataset = torchvision.datasets.SUN397(root='./data', split='test', download=True, transform=transform)
        num_classes = 397
    elif dataset_name == 'eurosat':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4308, 0.4308, 0.4308), (0.4303, 0.4303, 0.4303))
        ])
        train_dataset = torchvision.datasets.EuroSAT(root='./data', split='train', download=True, transform=transform)
        test_dataset = torchvision.datasets.EuroSAT(root='./data', split='test', download=True, transform=transform)
        num_classes = 10
    elif dataset_name == 'oxfordpets':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4308, 0.4308, 0.4308), (0.4303, 0.4303, 0.4303))
        ])
        train_dataset = torchvision.datasets.OxfordIIITPet(root='./data', split='train', download=True, transform=transform)
        test_dataset = torchvision.datasets.OxfordIIITPet(root='./data', split='test', download=True, transform=transform)
        num_classes = 37
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader, num_classes

def train_smm(model, train_loader, test_loader, num_classes, device, epochs=10, lr=0.01):
    """
    Train the SMM model with the specified dataset.
    """
    model.to(device)
    
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            # Apply SMM reprogramming
            reprogrammed_data = model(data)
            
            # Forward pass
            output = model.backbone(reprogrammed_data)
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        
        train_acc = 100 * correct / total
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                reprogrammed_data = model(data)
            val_output = model.backbone(reprogrammed_data)
            val_pred = val_output.argmax(dim=1)
            val_correct += val_pred.eq(target).sum().item()
            val_total += target.size(0)
        val_acc = 100 * val_correct / val_total
        print(f'Epoch [{epoch+1}/{epochs}], Val Acc: {val_acc:.2f}%')
    
    return model

def main():
    parser = argparse.ArgumentParser(description='Reproduce SMM Paper Results')
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset name')
    parser.add_argument('--model', type=str, default='resnet18', help='Model name')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load dataset
    print(f'Loading {args.dataset} dataset...')
    train_loader, test_loader, num_classes = load_dataset(args.dataset, args.batch_size)
    
    # Create SMM model
    print('Creating SMM model...')
    model = SMM(input_size=224, patch_size=8, num_layers=5)
    
    # Create classifier
    classifier = ReprogrammedClassifier(model_name=args.model, num_classes=num_classes)
    model.backbone = classifier.backbone
    
    # Train
    print('Training SMM model...')
    model = train_smm(model, train_loader, test_loader, num_classes, device, args.epochs)
    
    # Save model
    model_path = os.path.join(args.output_dir, f'smm_{args.dataset}_{args.model}_epoch_{args.epochs}.pth')
    torch.save(model.state_dict(), model_path)
    print(f'Model saved to {model_path}')
    
    # Save results
    results = {
        'dataset': args.dataset,
        'model': args.model,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'output_dir': args.output_dir
    }
    
    results_path = os.path.join(args.output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f'Results saved to {results_path}')
    
    print('Reproduction completed successfully!')

if __name__ == '__main__':
    main()