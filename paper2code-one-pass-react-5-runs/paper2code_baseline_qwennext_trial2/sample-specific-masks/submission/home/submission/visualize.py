#!/usr/bin/env python3
"""
Visualization script for SMM.
This script generates visualizations of the trained masks and reprogrammed images.
"""
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision
from PIL import Image
import json

# Add the SMM repository path
sys.path.append('/tmp/SMM')

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {device}")

class SMM(nn.Module):
    """
    Sample-specific Multi-channel Masks framework.
    This implements the SMM framework with visualization capabilities.
    """
    def __init__(self, model_name='resnet18', num_classes=10):
        super(SMM, self).__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Load the pre-trained model
        self.pretrained_model = self._load_pretrained_model()
        
        # Create the mask generator
        self.mask_generator = MaskGenerator(in_channels=3, num_layers=5)
        
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
        
        # Apply the noise pattern
        x_reprogrammed = x_resized + self.delta * mask
        
        # Apply the pre-trained model
        output = self.pretrained_model(x_resized)
        
        return output, mask

def main():
    """Main function to run the visualization script."""
    parser = argparse.ArgumentParser(description='Visualize SMM masks')
    parser.add_argument('--model', type=str, default='resnet18', help='Model to use: resnet18, resnet50, vit_b32')
    parser.add_argument('--dataset', type=str, default='flowers102', help='Dataset to use: cifar10, cifar100, svhn, gtsrb, flowers102, dtd, ucf101, food101, sun397, eurosat, oxfordpets')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--output_dir', type=str, default='/home/submission/results/visualizations', help='Output directory')
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
        dataset = torchvision.datasets.CIFAR10(root='/tmp/data', train=True, download=True, transform=transform)
        num_classes = 10
    elif args.dataset == 'cifar100':
        dataset = torchvision.datasets.CIFAR100(root='/tmp/data', train=True, download=True, transform=transform)
        num_classes = 100
    elif args.dataset == 'svhn':
        dataset = torchvision.datasets.SVHN(root='/tmp/data', split='train', download=True, transform=transform)
        num_classes = 10
    elif args.dataset == 'gtsrb':
        dataset = torchvision.datasets.GTSRB(root='/tmp/data', split='train', download=True, transform=transform)
        num_classes = 43
    elif args.dataset == 'flowers102':
        dataset = torchvision.datasets.Flowers102(root='/tmp/data', split='train', download=True, transform=transform)
        num_classes = 102
    elif args.dataset == 'dtd':
        dataset = torchvision.datasets.DTD(root='/tmp/data', split='train', download=True, transform=transform)
        num_classes = 47
    elif args.dataset == 'ucf101':
        dataset = torchvision.datasets.UCF101(root='/tmp/data', split='train', download=True, transform=transform)
        num_classes = 101
    elif args.dataset == 'food101':
        dataset = torchvision.datasets.Food101(root='/tmp/data', split='train', download=True, transform=transform)
        num_classes = 101
    elif args.dataset == 'sun397':
        dataset = torchvision.datasets.SUN397(root='/tmp/data', split='train', download=True, transform=transform)
        num_classes = 397
    elif args.dataset == 'eurosat':
        dataset = torchvision.datasets.EuroSAT(root='/tmp/data', download=True, transform=transform)
        num_classes = 10
    elif args.dataset == 'oxfordpets':
        dataset = torchvision.datasets.OxfordIIITPet(root='/tmp/data', split='train', download=True, transform=transform)
        num_classes = 37
    else:
        raise ValueError(f"Dataset {args.dataset} not supported')
    
    # Create data loader
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Create SMM model
    smm = SMM(model_name=args.model, num_classes=num_classes)
    
    # Train the model
    logging.info(f"Training SMM model {args.model} on dataset {args.dataset}")
    smm.train(train_loader, epochs=args.epochs, lr=args.lr)
    
    # Generate visualizations
    logging.info("Generating visualizations")
    
    # Get sample images
    data_iter = iter(data_loader)
    images, labels = next(data_iter)
    images = images.to(device)
    
    # Get predictions and masks
    predictions, masks = smm(images)
    
    # Plot images and masks
    fig, axes = plt.subplots(args.batch_size, 3, figsize=(15, 5 * args.batch_size))
    
    for i in range(args.batch_size):
        # Original image
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        img = np.clip(img, 0, 1)
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f'Original: {labels[i]}')
        axes[i, 0].axis('off')
        
        # Mask
        mask = masks[i].cpu().detach().numpy().transpose(1, 2, 0)
        mask = np.mean(mask, axis=2)
        axes[i, 1].imshow(mask, cmap='hot')
        axes[i, 1].set_title('Mask')
        axes[i, 1].axis('off')
        
        # Reprogrammed image
        reprogrammed = images[i].cpu().numpy().transpose(1, 2, 0)
        reprogrammed = np.clip(reprogrammed, 0, 1)
        axes[i, 2].imshow(reprogrammed)
        axes[i, 2].set_title('Reprogrammed')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'visualizations.png'))
    plt.close()
    
    logging.info(f"Visualizations saved to {os.path.join(args.output_dir, 'visualizations.png')}")
    
    logging.info("Visualization completed successfully!")

if __name__ == '__main__':
    main()