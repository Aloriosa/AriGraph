import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import random

# Import the model components from the linked code assets
from instance_model import AttributeNet, InstancewiseVisualPrompt

def set_seed(seed=42):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_dataset(dataset_name, image_size=224):
    """Load and preprocess the dataset"""
    transform_train = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if dataset_name == 'oxford_pets':
        train_dataset = torchvision.datasets.OxfordIIITPet(
            root='./data', split='trainval', download=True, transform=transform_train
        )
        test_dataset = torchvision.datasets.OxfordIIITPet(
            root='./data', split='test', download=True, transform=transform_test
        )
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    
    return train_dataset, test_dataset

def get_pretrained_model(model_name, num_classes):
    """Load a pre-trained model and freeze its parameters"""
    if model_name == 'resnet18':
        model = torchvision.models.resnet18(pretrained=True)
    elif model_name == 'resnet50':
        model = torchvision.models.resnet50(pretrained=True)
    elif model_name == 'vit_b_32':
        model = torchvision.models.vit_b_32(pretrained=True)
    else:
        raise ValueError(f"Model {model_name} not supported")
    
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace the final classifier layer
    if model_name.startswith('resnet'):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif model_name == 'vit_b_32':
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
    
    return model

def train_epoch(model, visual_prompt, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    visual_prompt.train()
    
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Apply visual reprogramming
        reprogrammed_images = visual_prompt(images)
        
        # Forward pass through frozen model
        outputs = model(reprogrammed_images)
        
        # Compute loss
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': loss.item(), 'acc': 100.*correct/total})
    
    return total_loss / len(dataloader), 100. * correct / total

def evaluate(model, visual_prompt, dataloader, criterion, device):
    """Evaluate the model"""
    model.eval()
    visual_prompt.eval()
    
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            # Apply visual reprogramming
            reprogrammed_images = visual_prompt(images)
            
            # Forward pass through frozen model
            outputs = model(reprogrammed_images)
            
            # Compute loss
            loss = criterion(outputs, labels)
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': loss.item(), 'acc': 100.*correct/total})
    
    return total_loss / len(dataloader), 100. * correct / total

def main():
    parser = argparse.ArgumentParser(description='Train Sample-Specific Masking for Visual Reprogramming')
    parser.add_argument('--dataset', type=str, default='oxford_pets', help='Dataset name')
    parser.add_argument('--model', type=str, default='resnet18', help='Pre-trained model name')
    parser.add_argument('--image_size', type=int, default=224, help='Input image size')
    parser.add_argument('--patch_size', type=int, default=8, help='Patch size for mask generation')
    parser.add_argument('--layers', type=int, default=5, help='Number of layers in mask generator')
    parser.add_argument('--channels', type=int, default=3, help='Number of channels in mask (3 for RGB)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory')
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading dataset...")
    train_dataset, test_dataset = get_dataset(args.dataset, args.image_size)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Get number of classes
    num_classes = len(train_dataset.classes)
    print(f"Number of classes: {num_classes}")
    
    # Load pre-trained model
    print(f"Loading pre-trained {args.model}...")
    pretrained_model = get_pretrained_model(args.model, num_classes)
    pretrained_model = pretrained_model.to(device)
    
    # Create visual prompting module
    print("Creating visual prompting module...")
    visual_prompt = InstancewiseVisualPrompt(
        size=args.image_size,
        layers=args.layers,
        patch_size=args.patch_size,
        channels=args.channels
    )
    visual_prompt = visual_prompt.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(visual_prompt.parameters(), lr=args.learning_rate)
    
    # Training loop
    print("Starting training...")
    best_acc = 0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(pretrained_model, visual_prompt, train_loader, criterion, optimizer, device)
        
        # Evaluate
        val_loss, val_acc = evaluate(pretrained_model, visual_prompt, test_loader, criterion, device)
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': pretrained_model.state_dict(),
                'visual_prompt_state_dict': visual_prompt.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss
            }, os.path.join(args.output_dir, 'best_model.pth'))
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% (Best: {best_acc:.2f}%)")
    
    print(f"\nTraining completed! Best validation accuracy: {best_acc:.2f}%")
    print(f"Model saved to: {os.path.join(args.output_dir, 'best_model.pth')}")

if __name__ == '__main__':
    main()