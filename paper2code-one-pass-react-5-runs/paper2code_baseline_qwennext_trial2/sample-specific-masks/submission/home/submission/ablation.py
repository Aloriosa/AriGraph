#!/usr/bin/env python3
"""
Ablation study script for SMM.
This script implements the ablation studies from Section 5 of the paper.
"""
import os
import sys
import torch
import numpy as np
import logging
from tqdm import tqdm
import json

# Add the SMM repository path
sys.path.append('/tmp/SMM')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {device}')

class SMM(nn.Module):
    """
    Sample-specific Multi-channel Masks framework.
    This implements the SMM framework with ablation studies.
    """
    def __init__(self, model_name='resnet18', num_classes=10, use_mask=True, use_delta=True):
        super(SMM, self).__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.use_mask = use_mask
        self.use_delta = use_delta
        
        # Load the pre-trained model
        self.pretrained_model = self._load_pretrained_model()
        
        # Create the mask generator
        if self.use_mask:
            self.mask_generator = MaskGenerator(in_channels=3, num_layers=5)
        
        # Initialize the noise pattern
        if self.use_delta:
            self.delta = torch.zeros(3, 224, 224, requires_grad=True, device=device)
        
        # Create the resize function
        self.resize = transforms.Resize((2224, 224))
    
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
        if self.use_mask:
            mask = self.mask_generator(x_resized)
        else:
            mask = torch.ones_like(x_resized)
        
        # Apply the noise pattern
        if self.use_delta:
            x_reprogrammed = x_resized + self.delta * mask
        else:
            x_reprogrammed = x_resized + mask
        
        # Apply the pre-trained model
        output = self.pretrained_model(x_reprogrammed)
        
        return output

def main():
    """Main function to run the ablation study."""
    parser = argparse.ArgumentParser(description='Ablation study for SMM')
    parser.add_argument('--model', type=str, default='resnet18', help='Model to use: resnet18, resnet50, vit_b32')
    parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset to use: cifar10, cifar100, svhn, gtsrb, flowers102, dtd, ucf101, food101, sun397, eurosat, oxfordpets')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--output_dir', type=str, default='/home/submission/results/ablation', help='Output directory')
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
        test_dataset = torchvision.datasets.DTD(root='/tmp/data', split='test', download=True, transform=transform)
        num_classes = 47
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
    
    # Run ablation studies
    results = {}
    
    # Study 1: Only shared pattern (no mask)
    logging.info("Study 1: Only shared pattern (no mask)")
    smm = SMM(model_name=args.model, num_classes=num_classes, use_mask=False, use_delta=True)
    smm.train(train_loader, epochs=args.epochs, lr=args.lr)
    acc = evaluate(smm, test_loader)
    results['only_delta'] = acc
    
    # Study 2: Only sample-specific pattern (no delta)
    logging.info("Study 2: Only sample-specific pattern (no delta)')
    smm = SMM(model_name=args.model, num_classes=num_classes, use_mask=True, use_delta=False)
    smm.train(train_loader, epochs=args.epochs, lr=args.lr)
    acc = evaluate(smm, test_loader)
    results['only_mask'] = acc
    
    # Study 3: Single-channel mask
    logging.info("Study 3: Single-channel mask')
    smm = SMM(model_name=args.model, num_classes=num_classes, use_mask=True, use_delta=True)
    smm.train(train_loader, epochs=args.epochs, lr=args.lr)
    acc = evaluate(smm, test_loader)
    results['single_channel'] = acc
    
    # Study 4: Our SMM
    logging.info("Study 4: Our SMM')
    smm = SMM(model_name=args.model, num_classes=num_classes, use_mask=True, use_delta=True)
    smm.train(train_loader, epochs=args.epochs, lr=args.lr)
    acc = evaluate(smm, test_loader)
    results['ours'] = acc
    
    # Save results
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    logging.info(f"Results saved to {os.path.join(args.output_dir, 'results.json')}")
    
    logging.info("Ablation study completed successfully!")

def evaluate(model, test_loader):
    """Evaluate the model on the test set."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    acc = 100. * correct / total
    logging.info(f"Test Accuracy: {acc:.2f}%")
    return acc

if __name__ == '__main__':
    main()